import pytest
import os
import tempfile
import gc
from app.storage.database import DatabaseManager
from app.storage.models import Lead, LeadStatus, AppointmentStatus, SentimentType
from app.tools.calendar_tool import CalendarTool
from app.tools.crm_tool import CRMTool
from app.intelligence.lead_qualifier import LeadQualifier
from app.intelligence.sentiment import SentimentAnalyzer


@pytest.fixture
def temp_db():
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


def test_appointment_booking_success(temp_db):
    """Verify successful booking when slot has no conflicts."""
    cal = CalendarTool(db=temp_db)
    lead = temp_db.save_lead(Lead(
        session_id="cal-sess-1",
        name="Elena Rostova",
        email="elena@enterprise.org"
    ))

    success, msg, apt = cal.book_appointment(
        session_id="cal-sess-1",
        start_time="2026-11-27T10:00:00Z",
        end_time="2026-11-27T10:30:00Z",
        attendee_name="Elena Rostova",
        attendee_email="elena@enterprise.org",
        lead_id=lead.id,
        notes="Interested in Enterprise Tier SLA."
    )

    assert success is True
    assert apt is not None
    assert apt.status == AppointmentStatus.CONFIRMED
    assert "https://meet.fortivex.ai/demo-" in apt.notes
    assert apt.lead_id == lead.id


def test_appointment_booking_slot_conflict(temp_db):
    """Verify that booking overlapping slots is rejected."""
    cal = CalendarTool(db=temp_db)

    # First booking
    s1, _, _ = cal.book_appointment(
        session_id="sess-conflict-1",
        start_time="2026-11-27T14:00:00Z",
        end_time="2026-11-27T14:30:00Z",
        attendee_name="User A",
        attendee_email="a@test.com"
    )
    assert s1 is True

    # Second booking overlapping with first (14:15 - 14:45)
    s2, msg, apt2 = cal.book_appointment(
        session_id="sess-conflict-2",
        start_time="2026-11-27T14:15:00Z",
        end_time="2026-11-27T14:45:00Z",
        attendee_name="User B",
        attendee_email="b@test.com"
    )
    assert s2 is False
    assert apt2 is None
    assert "already booked" in msg.lower()


def test_appointment_invalid_time_range(temp_db):
    """Verify rejection when end time is before or equal to start time."""
    cal = CalendarTool(db=temp_db)
    success, msg, apt = cal.book_appointment(
        session_id="sess-invalid",
        start_time="2026-11-27T15:00:00Z",
        end_time="2026-11-27T14:30:00Z",  # Invalid: before start
        attendee_name="Invalid User",
        attendee_email="invalid@test.com"
    )
    assert success is False
    assert apt is None
    assert "after start time" in msg.lower()


def test_crm_sync_success_mock(temp_db):
    """Verify CRM record creation and lead synchronization in Mock mode."""
    crm = CRMTool(db=temp_db)
    lead = temp_db.save_lead(Lead(
        session_id="crm-sess-1",
        name="Marcus Vance",
        email="marcus@globaltech.com",
        company="GlobalTech",
        role="Chief Technology Officer",
        budget="$25,000/year",
        authority="CTO",
        need="Inbound AI SDR to automate qualification",
        timeline="Immediate",
        lead_score=90,
        qualification_status=LeadStatus.HOT_LEAD
    ))

    cal = CalendarTool(db=temp_db)
    _, _, apt = cal.book_appointment(
        session_id="crm-sess-1",
        start_time="2026-11-28T16:00:00Z",
        end_time="2026-11-28T16:30:00Z",
        attendee_name="Marcus Vance",
        attendee_email="marcus@globaltech.com",
        lead_id=lead.id
    )

    success, msg, crm_record = crm.sync_lead(
        lead=lead,
        appointment=apt,
        notes="High-value prospect ready for enterprise rollout."
    )

    assert success is True
    assert crm_record.sync_status == "SYNCED"
    assert "MOCK-CRM-" in crm_record.external_id
    assert crm_record.payload["name"] == "Marcus Vance"
    assert crm_record.payload["appointment"]["start_time"] == "2026-11-28T16:00:00Z"

    # Query database to confirm persistence
    saved_records = temp_db.list_crm_records(lead_id=lead.id)
    assert len(saved_records) == 1
    assert saved_records[0].external_id == crm_record.external_id


def test_crm_sync_failure_handling(temp_db):
    """Verify that network or webhook failures are handled gracefully with FAILED status recorded."""
    crm = CRMTool(db=temp_db)
    crm.mode = "webhook"
    crm.webhook_url = "http://localhost:59999/unreachable-endpoint-that-fails"

    lead = temp_db.save_lead(Lead(
        session_id="crm-fail-sess",
        name="Failure Test User",
        email="fail@test.com"
    ))

    success, msg, record = crm.sync_lead(lead=lead)
    assert success is False
    assert record.sync_status == "FAILED"
    assert "error" in record.payload

    # Database still captured the failed sync attempt for audit trail
    records = temp_db.list_crm_records(lead_id=lead.id)
    assert len(records) == 1
    assert records[0].sync_status == "FAILED"


def test_complete_qualification_to_crm_workflow(temp_db):
    """
    Demonstrates a complete business workflow:
    1. Unqualified prospect interaction qualifies them to HOT_LEAD.
    2. Available slot is checked and booked via CalendarTool.
    3. Lead and appointment details are synchronized to CRM.
    """
    qualifier = LeadQualifier()
    analyzer = SentimentAnalyzer()
    cal = CalendarTool(db=temp_db)
    crm = CRMTool(db=temp_db)

    # Step 1: Customer Conversation & Qualification
    lead = Lead(session_id="e2e-workflow-session")
    utterance = (
        "I am Arthur Pendelton, Founder at Avalon AI. We have a budget of $20,000 "
        "and urgently need an autonomous voice sales agent to handle after-hours inbound inquiries this month."
    )
    sentiment = analyzer.analyze(utterance)
    lead = qualifier.evaluate_turn(lead, utterance, sentiment)
    lead.email = "arthur@avalon.ai"
    saved_lead = temp_db.save_lead(lead)

    assert saved_lead.qualification_status == LeadStatus.HOT_LEAD
    assert saved_lead.lead_score >= 80

    # Step 2: Calendar Slot Check & Booking
    start = "2026-11-29T11:00:00Z"
    end = "2026-11-29T11:30:00Z"
    assert cal.check_availability(start, end) is True

    book_ok, book_msg, apt = cal.book_appointment(
        session_id="e2e-workflow-session",
        start_time=start,
        end_time=end,
        attendee_name=saved_lead.name,
        attendee_email=saved_lead.email,
        lead_id=saved_lead.id,
        notes="Automated qualification completed via Voice Agent."
    )
    assert book_ok is True
    assert apt is not None

    # Step 3: CRM Synchronization
    crm_ok, crm_msg, crm_rec = crm.sync_lead(lead=saved_lead, appointment=apt)
    assert crm_ok is True
    assert crm_rec.sync_status == "SYNCED"

    # Step 4: Validate Database Records
    db_lead = temp_db.get_lead_by_session("e2e-workflow-session")
    db_apts = temp_db.get_appointments_by_lead(saved_lead.id)
    db_crm = temp_db.list_crm_records(saved_lead.id)

    assert db_lead.name == "Arthur Pendelton"
    assert len(db_apts) == 1
    assert db_apts[0].status == AppointmentStatus.CONFIRMED
    assert len(db_crm) == 1
    assert db_crm[0].payload["lead_score"] >= 80
