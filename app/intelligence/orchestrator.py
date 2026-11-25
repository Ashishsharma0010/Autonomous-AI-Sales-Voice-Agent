import base64
import json
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import httpx

from app.config import settings
from app.storage.database import DatabaseManager, get_db
from app.storage.models import Lead, LeadStatus, Appointment, SentimentType, CallTurn
from app.memory.buffer import ConversationMemory
from app.intelligence.sentiment import SentimentAnalyzer, SentimentResult
from app.intelligence.lead_qualifier import LeadQualifier
from app.intelligence.prompts import SALES_AGENT_SYSTEM_PROMPT
from app.rag.retriever import RAGRetriever, get_retriever, SearchResult
from app.tools.calendar_tool import CalendarTool
from app.tools.crm_tool import CRMTool
from app.audio.stt import SpeechToTextService, TranscriptionResult
from app.audio.tts import TextToSpeechService, SpeechResult
from app.audio.language_detector import LanguageDetector


class OrchestratorTurnResult(BaseModel):
    session_id: str
    user_text: str
    agent_text: str
    audio_base64: Optional[str] = None
    audio_content_type: str = "audio/mpeg"
    detected_language: str = "en"
    sentiment: SentimentResult
    lead: Lead
    rag_sources: List[Dict[str, Any]] = []
    appointment: Optional[Appointment] = None
    crm_synced: bool = False


class SalesAgentOrchestrator:
    """
    Central Sales Voice Agent Orchestrator.
    Coordinates the voice pipeline (STT/TTS), contextual memory, real-time sentiment analysis,
    BANT lead qualification, RAG retrieval, calendar booking, and CRM synchronization.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db()
        self.stt_service = SpeechToTextService()
        self.tts_service = TextToSpeechService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.lead_qualifier = LeadQualifier()
        self.retriever = get_retriever()
        self.calendar_tool = CalendarTool(db=self.db)
        self.crm_tool = CRMTool(db=self.db)
        self._sessions: Dict[str, ConversationMemory] = {}

    def get_or_create_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(session_id=session_id, db=self.db)
        return self._sessions[session_id]

    async def _generate_llm_response(
        self,
        memory: ConversationMemory,
        user_text: str,
        rag_context: str,
        lead: Lead,
        sentiment: SentimentResult,
        language: str = "en"
    ) -> str:
        """
        Generates contextual sales response using configured LLM provider,
        or deterministic consultative fallback if no external API key is configured.
        """
        # 1. External Groq LLM
        if settings.LLM_PROVIDER == "groq" and settings.is_groq_available:
            try:
                system_prompt = (
                    f"{SALES_AGENT_SYSTEM_PROMPT}\n\n"
                    f"KNOWLEDGE BASE CONTEXT (RAG):\n{rag_context}\n\n"
                    f"CUSTOMER PROFILE & BANT STATE:\n{memory.get_summary_context()}\n\n"
                    f"LANGUAGE DIRECTIVE: Reply in language code '{language}'. Keep response concise (1-2 sentences) for voice. "
                    f"ATTRIBUTION DIRECTIVE: Whenever citing metrics or claims (such as 80% cost reduction, <$0.10/call, or 40+ languages), explicitly state they are from 'our product knowledge base' or 'our product catalog'."
                )
                messages = [{"role": "system", "content": system_prompt}]
                for turn in memory.turns[-6:]:
                    role = "user" if turn.speaker == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
                messages.append({"role": "user", "content": user_text})

                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                        json={
                            "model": settings.LLM_MODEL,
                            "messages": messages,
                            "temperature": 0.3,
                            "max_tokens": 150
                        }
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        # 2. External OpenAI LLM
        if settings.LLM_PROVIDER == "openai" and settings.is_openai_available:
            try:
                system_prompt = (
                    f"{SALES_AGENT_SYSTEM_PROMPT}\n\n"
                    f"KNOWLEDGE BASE CONTEXT (RAG):\n{rag_context}\n\n"
                    f"CUSTOMER PROFILE & BANT STATE:\n{memory.get_summary_context()}\n\n"
                    f"LANGUAGE DIRECTIVE: Reply in language code '{language}'. Keep response concise (1-2 sentences) for voice. "
                    f"ATTRIBUTION DIRECTIVE: Whenever citing metrics or claims (such as 80% cost reduction, <$0.10/call, or 40+ languages), explicitly state they are from 'our product knowledge base' or 'our product catalog'."
                )
                messages = [{"role": "system", "content": system_prompt}]
                for turn in memory.turns[-6:]:
                    role = "user" if turn.speaker == "user" else "assistant"
                    messages.append({"role": role, "content": turn.text})
                messages.append({"role": "user", "content": user_text})

                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": messages,
                            "temperature": 0.3,
                            "max_tokens": 150
                        }
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        # 3. Consultative B2B Response Generator (Deterministic, zero-latency, offline-capable)
        # Formulates answers incorporating RAG grounding, customer name, and next qualification step
        greeting = f"Thanks {lead.name.split()[0]}, " if lead.name else "Thank you for reaching out. "

        # Handle disqualification
        if lead.qualification_status == LeadStatus.DISQUALIFIED:
            return "Understood. I will update our records and ensure we do not contact you again. Have a great day."

        # Handle objection grounded in RAG knowledge base
        if sentiment.is_objection and rag_context:
            return f"{greeting}I understand your concern. According to our published product knowledge base, customers report up to an 80% cost reduction compared to traditional human SDR overhead with 24/7 availability. Would you be open to exploring how this fits your timeline?"

        # Answer based on RAG knowledge
        if any(w in user_text.lower() for w in ["price", "pricing", "cost", "tier", "how much", "rate"]):
            return f"{greeting}Our Starter tier begins at $499 per month for 1,000 voice minutes, and Growth is $1,499 per month for 5,000 minutes. What monthly call volume are you planning for?"

        if "security" in user_text.lower() or "soc2" in user_text.lower() or "safe" in user_text.lower():
            return f"{greeting}Fortivex AI is fully SOC2 Type II certified and GDPR compliant, with all audio encrypted in transit and at rest using AES-256. What compliance requirements does your team have?"

        # Qualified / Hot Lead guiding to demo
        if lead.qualification_status in [LeadStatus.HOT_LEAD, LeadStatus.QUALIFIED]:
            return f"{greeting}Based on your requirements, Fortivex AI is an excellent fit for {lead.company or 'your team'}. I can reserve a 30-minute technical consultation and demo for you right now. Does tomorrow afternoon work?"

        # Progressive BANT qualification prompts
        if not lead.need:
            return f"{greeting}Fortivex AI provides autonomous inbound and outbound sales voice agents. What is your team's primary challenge with lead response times today?"
        elif not lead.timeline:
            return f"{greeting}That makes complete sense. How soon is your team looking to deploy an automated sales voice solution?"
        elif not lead.budget:
            return f"{greeting}Our tiers range from $499 to $1,499 per month. Does that align with what your organization has budgeted for sales automation?"
        else:
            return f"{greeting}Everything looks well aligned. I would love to schedule a live product demonstration for your team. Would this week work best for you?"

    async def process_turn(
        self,
        session_id: str,
        user_text: Optional[str] = None,
        audio_bytes: Optional[bytes] = None,
        language_override: Optional[str] = None
    ) -> OrchestratorTurnResult:
        """
        Executes one full conversation turn:
        1. STT audio transcription (if audio provided).
        2. Real-time sentiment & objection analysis.
        3. BANT lead evaluation and scoring.
        4. RAG knowledge chunk retrieval.
        5. Calendar & CRM tool execution if applicable.
        6. LLM response generation.
        7. Contextual memory update and DB sync.
        8. TTS voice synthesis in matched language.
        """
        memory = self.get_or_create_memory(session_id)
        current_lead = self.db.get_lead_by_session(session_id) or Lead(session_id=session_id)

        # 1. Speech-to-Text
        detected_language = language_override or "en"
        if audio_bytes and len(audio_bytes) > 0:
            stt_result = await self.stt_service.transcribe(audio_bytes, client_text=user_text)
            if stt_result.text and stt_result.text.strip():
                user_text = stt_result.text.strip()
            elif not user_text:
                user_text = ""
            detected_language = language_override or stt_result.language
        elif not user_text:
            user_text = ""

        # Language detection fallback on text
        if not language_override and user_text:
            detected_language = LanguageDetector.detect_language(user_text)

        # 2. Sentiment Analysis
        sentiment = self.sentiment_analyzer.analyze(user_text)

        # 3. BANT Lead Qualification
        updated_lead = self.lead_qualifier.evaluate_turn(current_lead, user_text, sentiment)
        saved_lead = self.db.save_lead(updated_lead)

        # Sync memory entities
        memory.entities.name = saved_lead.name
        memory.entities.email = saved_lead.email
        memory.entities.company = saved_lead.company
        memory.entities.role = saved_lead.role
        memory.entities.budget = saved_lead.budget
        memory.entities.authority = saved_lead.authority
        memory.entities.need = saved_lead.need
        memory.entities.timeline = saved_lead.timeline
        memory.entities.lead_score = saved_lead.lead_score
        memory.entities.qualification_status = saved_lead.qualification_status
        memory.entities.current_sentiment = sentiment.sentiment
        memory.entities.detected_language = detected_language

        # 4. RAG Retrieval
        rag_results = self.retriever.retrieve(user_text, top_k=2)
        rag_context = self.retriever.format_for_prompt(rag_results)
        rag_sources_meta = [
            {"id": r.document.id, "title": r.document.title, "category": r.document.category, "score": r.score}
            for r in rag_results
        ]

        # 5. Calendar / CRM Tool Actions
        appointment: Optional[Appointment] = None
        crm_synced: bool = False

        # If user explicitly requests a booking or says "yes/schedule/book" when qualified
        booking_intent = any(w in user_text.lower() for w in ["book", "schedule", "reserve", "confirm demo", "yes please", "tomorrow", "calendar"])
        if booking_intent and saved_lead.qualification_status in [LeadStatus.QUALIFIED, LeadStatus.HOT_LEAD]:
            # Dynamically compute upcoming open slots
            from datetime import timedelta, timezone, datetime
            now_dt = datetime.now(timezone.utc)
            candidate_slots = []
            for d in range(1, 20):
                base_day = (now_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                for hour in [10, 11, 14, 15, 16]:
                    candidate_slots.append((f"{base_day}T{hour:02d}:00:00Z", f"{base_day}T{hour:02d}:30:00Z"))

            for start_slot, end_slot in candidate_slots:
                if self.calendar_tool.check_availability(start_slot, end_slot):
                    success, _, apt = self.calendar_tool.book_appointment(
                        session_id=session_id,
                        start_time=start_slot,
                        end_time=end_slot,
                        attendee_name=saved_lead.name or "Prospect",
                        attendee_email=saved_lead.email or "prospect@client.com",
                        lead_id=saved_lead.id,
                        notes=f"Auto-booked demo. Lead score: {saved_lead.lead_score}"
                    )
                    if success:
                        appointment = apt
                        break

            # Sync to CRM
            crm_ok, _, _ = self.crm_tool.sync_lead(lead=saved_lead, appointment=appointment)
            crm_synced = crm_ok

        # 6. LLM Response Generation
        agent_text = await self._generate_llm_response(
            memory=memory,
            user_text=user_text,
            rag_context=rag_context,
            lead=saved_lead,
            sentiment=sentiment,
            language=detected_language
        )

        if appointment:
            agent_text += f" I have confirmed your demonstration for {appointment.start_time}. A calendar invitation has been sent."

        # 7. Record turns in contextual memory
        memory.add_turn(speaker="user", text=user_text, sentiment=sentiment.sentiment, language=detected_language)
        memory.add_turn(speaker="agent", text=agent_text, language=detected_language)
        memory.sync_to_db()

        # 8. Text-to-Speech Synthesis
        tts_result = await self.tts_service.synthesize(text=agent_text, language=detected_language)
        audio_b64 = base64.b64encode(tts_result.audio_bytes).decode("utf-8") if tts_result.audio_bytes else None

        return OrchestratorTurnResult(
            session_id=session_id,
            user_text=user_text,
            agent_text=agent_text,
            audio_base64=audio_b64,
            audio_content_type=tts_result.content_type,
            detected_language=detected_language,
            sentiment=sentiment,
            lead=saved_lead,
            rag_sources=rag_sources_meta,
            appointment=appointment,
            crm_synced=crm_synced
        )
