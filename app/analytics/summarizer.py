from typing import List, Optional
from pydantic import BaseModel, Field
import httpx
from app.config import settings
from app.storage.models import CallTurn, Lead


class SummaryResult(BaseModel):
    summary: str
    key_takeaways: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)


class CallSummarizer:
    """
    Call Summarisation Engine.
    Generates an executive summary, key takeaways, and action items
    from sales dialogue transcripts.
    """

    async def summarize_call(
        self,
        turns: List[CallTurn],
        lead: Optional[Lead] = None
    ) -> SummaryResult:
        """
        Synthesizes conversation transcript into structured summary,
        key customer requirements, and follow-up action items.
        """
        if not turns:
            return SummaryResult(
                summary="Call completed with no conversational dialogue recorded.",
                key_takeaways=["Call initiated without customer interaction."],
                action_items=["Verify customer contact details and initiate follow-up outreach."]
            )

        transcript_text = "\n".join([
            f"{'Customer' if t.speaker == 'user' else 'Sales Agent'}: {t.text}"
            for t in turns
        ])

        # 1. External LLM Summarization (if API key configured)
        if (settings.is_groq_available or settings.is_openai_available):
            prompt = (
                f"You are a sales operations analyst. Analyze this sales conversation transcript and provide a JSON response:\n\n"
                f"TRANSCRIPT:\n{transcript_text}\n\n"
                f"LEAD INFO:\nName: {lead.name if lead else 'Unknown'}, Company: {lead.company if lead else 'Unknown'}, "
                f"Role: {lead.role if lead else 'Unknown'}, Budget: {lead.budget if lead else 'None'}\n\n"
                f"Return ONLY valid JSON in this exact format:\n"
                f"{{\n"
                f'  "summary": "2-3 sentence executive summary of the conversation.",\n'
                f'  "key_takeaways": ["point 1", "point 2", "point 3"],\n'
                f'  "action_items": ["item 1", "item 2"]\n'
                f"}}"
            )
            try:
                if settings.is_groq_available:
                    async with httpx.AsyncClient(timeout=8.0) as client:
                        resp = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                            json={
                                "model": settings.LLM_MODEL,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.2,
                                "response_format": {"type": "json_object"}
                            }
                        )
                        if resp.status_code == 200:
                            import json
                            data = json.loads(resp.json()["choices"][0]["message"]["content"])
                            return SummaryResult(**data)
            except Exception:
                pass  # Fall through to deterministic summarizer

        # 2. Consultative Analytical Summarizer (Deterministic, offline-reliable)
        prospect_name = lead.name if lead and lead.name else "Prospect"
        company = lead.company if lead and lead.company else "Organization"
        user_turns = [t for t in turns if t.speaker == "user"]

        takeaways = []
        action_items = []

        if lead and lead.need:
            takeaways.append(f"Identified primary requirement: {lead.need}")
        if lead and lead.budget:
            takeaways.append(f"Budget indicated: {lead.budget}")
        if lead and lead.timeline:
            takeaways.append(f"Target deployment window: {lead.timeline}")
        if not takeaways:
            takeaways.append(f"{prospect_name} engaged in product discovery regarding Fortivex AI sales automation.")

        # Determine action items based on qualification state
        if lead and lead.qualification_status.value == "HOT_LEAD":
            action_items.append("Prepare tailored technical demo and environment walkthrough.")
            action_items.append("Review prospective calendar invite and assign Account Executive.")
        elif lead and lead.qualification_status.value == "QUALIFIED":
            action_items.append("Send product pricing tier specifications and customer case studies.")
            action_items.append("Follow up within 48 hours to confirm decision maker timeline.")
        elif lead and lead.qualification_status.value == "DISQUALIFIED":
            action_items.append("Mark contact as unqualified in CRM and pause automated outreach.")
        else:
            action_items.append("Conduct follow-up discovery call to clarify BANT requirements.")

        summary = (
            f"Sales conversation conducted with {prospect_name} from {company}. "
            f"The customer discussed {'sales automation requirements' if not lead or not lead.need else lead.need}. "
            f"Lead qualification status concluded as {lead.qualification_status.value if lead else 'UNQUALIFIED'} "
            f"with a BANT score of {lead.lead_score if lead else 0}/100."
        )

        return SummaryResult(
            summary=summary,
            key_takeaways=takeaways,
            action_items=action_items
        )
