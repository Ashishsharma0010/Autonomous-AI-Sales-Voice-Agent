from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from app.storage.models import SentimentType, LeadStatus


class SessionEntityStore(BaseModel):
    """Stores extracted conversation entities and user attributes for an active session."""
    session_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    budget: Optional[str] = None
    authority: Optional[str] = None
    need: Optional[str] = None
    timeline: Optional[str] = None
    lead_score: int = 0
    qualification_status: LeadStatus = LeadStatus.UNQUALIFIED
    current_sentiment: SentimentType = SentimentType.NEUTRAL
    detected_language: str = "en"
    detected_objections: list[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)
