import pytest
import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.storage.database import DatabaseManager
from app.storage.models import Lead, LeadStatus, Appointment, AppointmentStatus, CallTurn, SentimentType
from app.analytics.summarizer import CallSummarizer
from app.analytics.insights import BusinessInsightExtractor


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_call_summarizer_with_dialogue():
    """Verify that CallSummarizer extracts concise summary, takeaways, and action items."""
    summarizer = CallSummarizer()
    lead = Lead(
        session_id="sum-sess-1",
        name="Samantha Reed",
        company="NexGen AI",
        role="VP of Sales",
        budget="$15,000/yr",
        need="Automate after-hours inbound SDR calls",
        timeline="Next month",
        lead_score=85,
        qualification_status=LeadStatus.HOT_LEAD
    )
    turns = [
        CallTurn(speaker="agent", text="Welcome to Fortivex AI. How can I help?"),
        CallTurn(speaker="user", text="I am Samantha Reed from NexGen AI. We need an autonomous inbound sales agent.", sentiment=SentimentType.POSITIVE),
        CallTurn(speaker="agent", text="Fortivex Inbound SDR answers calls 24/7. What is your budget and timeline?"),
        CallTurn(speaker="user", text="We have $15,000 budgeted and need to deploy next month.", sentiment=SentimentType.POSITIVE),
        CallTurn(speaker="agent", text="Confirmed your demonstration for tomorrow. Look forward to speaking.")
    ]

    res = await summarizer.summarize_call(turns, lead)

    assert "Samantha Reed" in res.summary
    assert "NexGen AI" in res.summary
    assert len(res.key_takeaways) >= 2
    assert any("budget" in t.lower() or "$15,000" in t for t in res.key_takeaways)
    assert len(res.action_items) >= 1


@pytest.mark.asyncio
async def test_call_summarizer_empty_dialogue():
    """Verify graceful handling when call ends without dialogue turns."""
    summarizer = CallSummarizer()
    res = await summarizer.summarize_call([], None)

    assert "no conversational dialogue" in res.summary.lower()
    assert len(res.key_takeaways) > 0
    assert len(res.action_items) > 0


@pytest.mark.asyncio
async def test_business_insight_extractor():
    """Verify extraction of sentiment trajectory, objections, conversion probability, and BANT."""
    extractor = BusinessInsightExtractor()
    sess_id = f"insight-{uuid.uuid4().hex[:8]}"

    lead = Lead(
        session_id=sess_id,
        name="Victor Vance",
        company="Vance Dynamics",
        role="Director of Growth",
        budget="$20,000",
        authority="Director of Growth",
        need="Inbound call qualification",
        timeline="Q1",
        lead_score=85,
        qualification_status=LeadStatus.HOT_LEAD
    )
    apt = Appointment(
        session_id=sess_id,
        title="Fortivex AI Demo",
        start_time="2026-11-30T15:00:00Z",
        end_time="2026-11-30T15:30:00Z",
        status=AppointmentStatus.CONFIRMED
    )
    turns = [
        CallTurn(speaker="agent", text="Hello Victor, how can I help?"),
        CallTurn(speaker="user", text="Is your solution expensive compared to human SDRs?", sentiment=SentimentType.HESITANT),
        CallTurn(speaker="agent", text="According to our knowledge base, customers report up to 80% cost reduction."),
        CallTurn(speaker="user", text="That sounds great, let's schedule a demo for this month.", sentiment=SentimentType.POSITIVE)
    ]

    call_log = await extractor.generate_call_analytics(
        session_id=sess_id,
        turns=turns,
        lead=lead,
        appointment=apt,
        duration_seconds=180
    )

    assert call_log.session_id == sess_id
    assert call_log.duration_seconds == 180
    assert "HESITANT" in call_log.sentiment_trajectory
    assert "POSITIVE" in call_log.sentiment_trajectory
    assert "budget_objection" in call_log.detected_objections

    insights = call_log.sales_insights
    assert insights["conversion_probability_percent"] >= 70
    assert insights["appointment_scheduled"] is True
    assert insights["bant_breakdown"]["total_score"] == 85
    assert insights["bant_breakdown"]["budget_confirmed"] is True


def test_analytics_api_endpoints(client):
    """Verify REST API endpoints: /api/session/{session_id}/end, /analytics, and /analytics/overview."""
    sess_id = f"api-sess-{uuid.uuid4().hex[:8]}"

    # 1. Chat turn to populate session memory
    client.post("/api/chat", json={
        "session_id": sess_id,
        "text": "Hello, I am Rachel Green, Head of Sales at AlphaCorp. We have a budget of $5,000 and want to launch in Q1."
    })

    # 2. End call session
    resp_end = client.post(f"/api/session/{sess_id}/end?duration_seconds=145")
    assert resp_end.status_code == 200
    data_end = resp_end.json()
    assert data_end["session_id"] == sess_id
    assert data_end["duration_seconds"] == 145
    assert len(data_end["summary"]) > 0
    assert "sales_insights" in data_end

    # 3. Retrieve analytics
    resp_get = client.get(f"/api/session/{sess_id}/analytics")
    assert resp_get.status_code == 200
    data_get = resp_get.json()
    assert data_get["session_id"] == sess_id
    assert "AlphaCorp" in data_get["summary"] or "Rachel" in data_get["summary"]

    # 4. Analytics overview
    resp_overview = client.get("/api/analytics/overview")
    assert resp_overview.status_code == 200
    overview = resp_overview.json()
    assert overview["total_prospects"] >= 1
    assert "conversion_pipeline_rate" in overview
