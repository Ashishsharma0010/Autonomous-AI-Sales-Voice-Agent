from typing import List, Optional
from app.storage.models import CallTurn, SentimentType, Lead
from app.storage.database import DatabaseManager, get_db
from app.memory.entity_store import SessionEntityStore


class ConversationMemory:
    """
    Contextual Memory Engine for Sales Voice Conversations.
    Maintains a sliding window of conversational dialogue turns, manages session state,
    and synchronizes entity state with SQLite persistence.
    """

    def __init__(self, session_id: str, max_turns: int = 15, db: Optional[DatabaseManager] = None):
        self.session_id = session_id
        self.max_turns = max_turns
        self.db = db or get_db()
        self.turns: List[CallTurn] = []
        self.entities = SessionEntityStore(session_id=session_id)
        self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Hydrates previous lead state if existing session found in database."""
        lead = self.db.get_lead_by_session(self.session_id)
        if lead:
            self.entities.name = lead.name
            self.entities.email = lead.email
            self.entities.company = lead.company
            self.entities.role = lead.role
            self.entities.budget = lead.budget
            self.entities.authority = lead.authority
            self.entities.need = lead.need
            self.entities.timeline = lead.timeline
            self.entities.lead_score = lead.lead_score
            self.entities.qualification_status = lead.qualification_status

    def add_turn(
        self,
        speaker: str,
        text: str,
        sentiment: Optional[SentimentType] = None,
        language: str = "en"
    ) -> CallTurn:
        """Adds a turn to memory, updates current sentiment, and enforces sliding window."""
        turn = CallTurn(
            speaker=speaker,
            text=text,
            sentiment=sentiment,
            detected_language=language
        )
        self.turns.append(turn)

        if sentiment:
            self.entities.current_sentiment = sentiment
        if language:
            self.entities.detected_language = language

        # Enforce sliding window to stay within LLM context window
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

        return turn

    def get_formatted_history(self, last_n: Optional[int] = None) -> str:
        """Formats conversational turns into clean script transcript for LLM prompts."""
        selected_turns = self.turns if last_n is None else self.turns[-last_n:]
        lines = []
        for t in selected_turns:
            label = "Customer" if t.speaker == "user" else "Sales Agent"
            lines.append(f"{label}: {t.text}")
        return "\n".join(lines)

    def sync_to_db(self) -> Lead:
        """Persists current entity and lead qualification state into SQLite database."""
        lead = Lead(
            session_id=self.session_id,
            name=self.entities.name,
            email=self.entities.email,
            company=self.entities.company,
            role=self.entities.role,
            budget=self.entities.budget,
            authority=self.entities.authority,
            need=self.entities.need,
            timeline=self.entities.timeline,
            lead_score=self.entities.lead_score,
            qualification_status=self.entities.qualification_status,
            notes=f"Current sentiment: {self.entities.current_sentiment.value}"
        )
        return self.db.save_lead(lead)

    def get_summary_context(self) -> str:
        """Returns concise summary of currently known customer attributes."""
        e = self.entities
        return (
            f"Customer Name: {e.name or 'Unknown'} | Company: {e.company or 'Unknown'} | Role: {e.role or 'Unknown'}\n"
            f"BANT Status - Budget: {e.budget or 'Pending'}, Authority: {e.authority or 'Pending'}, "
            f"Need: {e.need or 'Pending'}, Timeline: {e.timeline or 'Pending'}\n"
            f"Lead Score: {e.lead_score}/100 | Qualification: {e.qualification_status.value} | Sentiment: {e.current_sentiment.value}"
        )
