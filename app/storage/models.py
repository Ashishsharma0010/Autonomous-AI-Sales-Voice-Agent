from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LeadStatus(str, Enum):
    UNQUALIFIED = "UNQUALIFIED"
    QUALIFYING = "QUALIFYING"
    QUALIFIED = "QUALIFIED"
    HOT_LEAD = "HOT_LEAD"
    DISQUALIFIED = "DISQUALIFIED"


class SentimentType(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    HESITANT = "HESITANT"
    NEGATIVE = "NEGATIVE"
    FRUSTRATED = "FRUSTRATED"


class AppointmentStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class Lead(BaseModel):
    id: Optional[str] = None
    session_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    budget: Optional[str] = None
    authority: Optional[str] = None
    need: Optional[str] = None
    timeline: Optional[str] = None
    lead_score: int = Field(default=0, ge=0, le=100)
    qualification_status: LeadStatus = LeadStatus.UNQUALIFIED
    notes: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)


class Appointment(BaseModel):
    id: Optional[str] = None
    session_id: str
    lead_id: Optional[str] = None
    title: str = "Fortivex AI Product Demo & Technical Consultation"
    start_time: str
    end_time: str
    timezone: str = "UTC"
    attendee_name: Optional[str] = None
    attendee_email: Optional[str] = None
    status: AppointmentStatus = AppointmentStatus.CONFIRMED
    notes: Optional[str] = None
    created_at: str = Field(default_factory=_utc_now_iso)


class CallTurn(BaseModel):
    speaker: str  # "agent" | "user"
    text: str
    timestamp: str = Field(default_factory=_utc_now_iso)
    sentiment: Optional[SentimentType] = None
    detected_language: Optional[str] = "en"


class CallLog(BaseModel):
    id: Optional[str] = None
    session_id: str
    lead_id: Optional[str] = None
    transcript: List[CallTurn] = Field(default_factory=list)
    summary: Optional[str] = None
    key_takeaways: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    sentiment_trajectory: List[str] = Field(default_factory=list)
    detected_objections: List[str] = Field(default_factory=list)
    sales_insights: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: int = 0
    created_at: str = Field(default_factory=_utc_now_iso)


class CRMRecord(BaseModel):
    id: Optional[str] = None
    lead_id: str
    crm_system: str = "mock"  # "mock" | "hubspot" | "salesforce" | "webhook"
    external_id: Optional[str] = None
    sync_status: str = "SYNCED"  # "PENDING" | "SYNCED" | "FAILED"
    payload: Dict[str, Any] = Field(default_factory=dict)
    synced_at: str = Field(default_factory=_utc_now_iso)
