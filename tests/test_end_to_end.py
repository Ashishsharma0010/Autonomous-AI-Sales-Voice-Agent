import pytest
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.storage.database import get_db
from app.storage.models import LeadStatus, AppointmentStatus


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_complete_end_to_end_business_workflow(client):
    """
    Complete End-to-End Technical & Commercial Demonstration:
    Step 1: Customer initiates voice session & introduces need.
    Step 2: Customer queries pricing -> RAG knowledge retrieved.
    Step 3: Customer raises pricing objection -> RAG objection resolution retrieved.
    Step 4: Customer confirms Budget & Timeline -> BANT reaches HOT_LEAD.
    Step 5: Automated appointment booking triggered & confirmed with meeting link.
    Step 6: CRM synchronization records lead, appointment, and lead score.
    Step 7: Session finalized via /api/session/{id}/end.
    Step 8: Post-call executive summary, key takeaways, and action items generated.
    Step 9: Business insights extracted (conversion probability, sentiment trend, BANT audit).
    """
    session_id = f"e2e-demo-{uuid.uuid4().hex[:8]}"

    # ==================== STEP 1: INITIAL DISCOVERY ====================
    t1_payload = {
        "session_id": session_id,
        "text": "Hi, I am Marcus Vance, Chief Technology Officer at GlobalTech. We urgently need an autonomous voice sales agent to handle after-hours inbound inquiries."
    }
    r1 = client.post("/api/chat", json=t1_payload)
    assert r1.status_code == 200
    d1 = r1.json()

    assert d1["lead"]["name"] == "Marcus Vance"
    assert d1["lead"]["company"] == "GlobalTech"
    assert d1["lead"]["role"] == "Chief Technology Officer"
    assert d1["lead"]["lead_score"] >= 50
    assert d1["lead"]["qualification_status"] in ["QUALIFIED", "HOT_LEAD"]
    assert d1["audio_base64"] is not None  # TTS audio synthesized

    # ==================== STEP 2: RAG PRICING INQUIRY ====================
    t2_payload = {
        "session_id": session_id,
        "text": "What is the pricing for your Starter and Growth tiers?"
    }
    r2 = client.post("/api/chat", json=t2_payload)
    assert r2.status_code == 200
    d2 = r2.json()

    assert len(d2["rag_sources"]) > 0
    assert any("pricing" in str(s).lower() or "prod-003" in str(s) for s in d2["rag_sources"])
    assert "$499" in d2["agent_text"] or "Starter" in d2["agent_text"]

    # ==================== STEP 3: OBJECTION HANDLING ====================
    t3_payload = {
        "session_id": session_id,
        "text": "Your pricing seems too expensive compared to traditional human SDRs."
    }
    r3 = client.post("/api/chat", json=t3_payload)
    assert r3.status_code == 200
    d3 = r3.json()

    assert d3["sentiment"]["is_objection"] is True
    assert "80%" in d3["agent_text"]
    assert "knowledge base" in d3["agent_text"].lower() or "product catalog" in d3["agent_text"].lower()

    # ==================== STEP 4 & 5: BANT QUALIFICATION & DEMO BOOKING ====================
    t4_payload = {
        "session_id": session_id,
        "text": "We have an approved budget of $20,000 and want to deploy in Q1. Yes please, let's schedule the demo."
    }
    r4 = client.post("/api/chat", json=t4_payload)
    assert r4.status_code == 200
    d4 = r4.json()

    # BANT Score must reach HOT_LEAD
    assert d4["lead"]["lead_score"] >= 75
    assert d4["lead"]["qualification_status"] == "HOT_LEAD"
    assert d4["lead"]["budget"] is not None
    assert d4["lead"]["timeline"] is not None

    # Automated Appointment Booking verified
    assert d4["appointment"] is not None
    assert d4["appointment"]["status"] == "CONFIRMED"
    assert "https://meet.fortivex.ai/demo-" in d4["appointment"]["notes"]
    assert "confirmed your demonstration" in d4["agent_text"].lower()

    # CRM Synchronization verified
    assert d4["crm_synced"] is True

    # ==================== STEP 6: VERIFY DATABASE PERSISTENCE ====================
    db = get_db()
    saved_lead = db.get_lead_by_session(session_id)
    assert saved_lead is not None
    assert saved_lead.name == "Marcus Vance"
    assert saved_lead.lead_score >= 75

    saved_apts = db.get_appointments_by_lead(saved_lead.id)
    assert len(saved_apts) == 1
    assert saved_apts[0].status == AppointmentStatus.CONFIRMED

    saved_crm = db.list_crm_records(lead_id=saved_lead.id)
    assert len(saved_crm) >= 1
    assert saved_crm[0].sync_status == "SYNCED"

    # ==================== STEP 7 & 8: FINALIZE CALL & GENERATE SUMMARY ====================
    r_end = client.post(f"/api/session/{session_id}/end?duration_seconds=210")
    assert r_end.status_code == 200
    d_end = r_end.json()

    assert d_end["session_id"] == session_id
    assert d_end["duration_seconds"] == 210
    assert "Marcus Vance" in d_end["summary"]
    assert "GlobalTech" in d_end["summary"]
    assert len(d_end["key_takeaways"]) >= 2
    assert len(d_end["action_items"]) >= 1

    # ==================== STEP 9: BUSINESS INSIGHTS VERIFICATION ====================
    insights = d_end["sales_insights"]
    assert insights["conversion_probability_percent"] >= 70
    assert insights["lead_fit"] in ["EXCELLENT", "GOOD"]
    assert insights["appointment_scheduled"] is True
    assert "budget_objection" in insights["primary_objections"]
    assert insights["bant_breakdown"]["total_score"] >= 75
    assert "commercial_recommendation" in insights

    # Verify retrieval via REST analytics endpoint
    r_analytics = client.get(f"/api/session/{session_id}/analytics")
    assert r_analytics.status_code == 200
    d_analytics = r_analytics.json()
    assert d_analytics["summary"] == d_end["summary"]
