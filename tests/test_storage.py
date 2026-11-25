import os
import gc
import tempfile
import pytest
from app.config import Settings
from app.storage.database import DatabaseManager
from app.storage.models import (
    Lead, Appointment, CallLog, CRMRecord,
    LeadStatus, SentimentType, AppointmentStatus, CallTurn
)


@pytest.fixture
def temp_db():
    """Provides a fresh temporary SQLite database for test isolation."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    db.init_db()
    yield db
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_settings_load():
    """Verify settings defaults and provider detection."""
    settings = Settings(
        GROQ_API_KEY="gsk_test123",
        OPENAI_API_KEY="",
        TTS_PROVIDER="edge"
    )
    assert settings.is_groq_available is True
    assert settings.is_openai_available is False
    assert settings.TTS_PROVIDER == "edge"
    assert settings.CALENDAR_MODE == "mock"


def test_database_initialization(temp_db):
    """Verify all 4 core tables exist upon database initialization."""
    with temp_db.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = [t["name"] for t in tables]

    assert "leads" in table_names
    assert "appointments" in table_names
    assert "call_logs" in table_names
    assert "crm_records" in table_names


def test_lead_lifecycle(temp_db):
    """Verify Lead insertion, BANT score update, and retrieval by session."""
    initial_lead = Lead(
        session_id="session-001",
        name="Alex Smith",
        email="alex@enterprise.com",
        company="Enterprise Corp",
        role="VP Engineering",
        budget="$50,000/yr",
        authority="Decision Maker",
        need="Autonomous sales voice outreach",
        timeline="Q1",
        lead_score=40,
        qualification_status=LeadStatus.QUALIFYING
    )
    saved = temp_db.save_lead(initial_lead)
    assert saved.id is not None
    assert saved.lead_score == 40
    assert saved.qualification_status == LeadStatus.QUALIFYING

    # Update lead score and qualification status
    updated_lead = Lead(
        session_id="session-001",
        name="Alex Smith",
        lead_score=85,
        qualification_status=LeadStatus.HOT_LEAD,
        notes="High interest in multi-lingual enterprise deployment."
    )
    res = temp_db.save_lead(updated_lead)
    assert res.lead_score == 85
    assert res.qualification_status == LeadStatus.HOT_LEAD
    assert res.company == "Enterprise Corp"  # Preserved from initial lead via COALESCE

    # Retrieve lead by session
    fetched = temp_db.get_lead_by_session("session-001")
    assert fetched is not None
    assert fetched.name == "Alex Smith"
    assert fetched.lead_score == 85


def test_appointment_booking_and_conflict_detection(temp_db):
    """Verify appointment creation and time conflict detection."""
    lead = temp_db.save_lead(Lead(
        session_id="session-002",
        name="Sarah Connor",
        email="sarah@tech.co"
    ))

    apt1 = Appointment(
        session_id="session-002",
        lead_id=lead.id,
        title="Fortivex AI Demo",
        start_time="2026-11-26T14:00:00Z",
        end_time="2026-11-26T14:30:00Z",
        attendee_name="Sarah Connor",
        attendee_email="sarah@tech.co",
        status=AppointmentStatus.CONFIRMED
    )
    saved_apt = temp_db.save_appointment(apt1)
    assert saved_apt.id is not None

    # Overlapping slot check: should return True for conflict
    has_conflict = temp_db.check_slot_conflict("2026-11-26T14:15:00Z", "2026-11-26T14:45:00Z")
    assert has_conflict is True

    # Non-overlapping slot check: should return False
    no_conflict = temp_db.check_slot_conflict("2026-11-26T15:00:00Z", "2026-11-26T15:30:00Z")
    assert no_conflict is False


def test_call_log_persistence(temp_db):
    """Verify call transcript, sentiment trajectory, and executive summary persistence."""
    turns = [
        CallTurn(speaker="agent", text="Hello! How can I help you today?"),
        CallTurn(speaker="user", text="We need an AI agent for inbound sales qualification.", sentiment=SentimentType.POSITIVE),
        CallTurn(speaker="agent", text="Fortivex AI can qualify your leads in real-time.", sentiment=SentimentType.POSITIVE)
    ]

    call_log = CallLog(
        session_id="session-003",
        transcript=turns,
        summary="Customer inquired about real-time lead qualification.",
        key_takeaways=["Needs inbound voice automation", "Interested in qualification speed"],
        action_items=["Send pricing tier documentation", "Schedule 30-min deep dive demo"],
        sentiment_trajectory=["POSITIVE", "POSITIVE"],
        detected_objections=[],
        sales_insights={"lead_fit": "high", "urgency": "medium"},
        duration_seconds=120
    )

    saved_log = temp_db.save_call_log(call_log)
    assert saved_log.id is not None

    retrieved = temp_db.get_call_log_by_session("session-003")
    assert retrieved is not None
    assert len(retrieved.transcript) == 3
    assert retrieved.transcript[1].speaker == "user"
    assert retrieved.transcript[1].sentiment == SentimentType.POSITIVE
    assert "Send pricing tier documentation" in retrieved.action_items


def test_crm_record_sync(temp_db):
    """Verify CRM record logging and payload serialization."""
    lead = temp_db.save_lead(Lead(
        session_id="session-004",
        name="Jordan Lee",
        email="jordan@innovate.io",
        company="Innovate AI"
    ))

    crm_record = CRMRecord(
        lead_id=lead.id,
        crm_system="mock",
        external_id="MOCK-CRM-992",
        sync_status="SYNCED",
        payload={"contact_id": "MOCK-CRM-992", "score": 75, "status": "Hot Lead"}
    )

    saved = temp_db.save_crm_record(crm_record)
    assert saved.id is not None

    records = temp_db.list_crm_records(lead_id=lead.id)
    assert len(records) == 1
    assert records[0].external_id == "MOCK-CRM-992"
    assert records[0].payload["score"] == 75
