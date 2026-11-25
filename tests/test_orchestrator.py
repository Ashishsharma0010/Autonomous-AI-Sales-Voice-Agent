import pytest
import json
import base64
import uuid
from fastapi.testclient import TestClient

from app.main import app, orchestrator
from app.storage.database import get_db
from app.storage.models import LeadStatus, SentimentType


@pytest.fixture
def client():
    """Provides a TestClient for FastAPI endpoints and WebSockets."""
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    """Verify health check endpoint returns 200 and provider info."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


@pytest.mark.asyncio
async def test_orchestrator_text_turn():
    """Verify orchestrator processes text turn, retrieves RAG context, and synthesizes audio."""
    session_id = "orch-test-text-1"
    user_query = "Hi, my name is Linda from FinTech Solutions. What is the price of the Starter tier?"

    result = await orchestrator.process_turn(session_id=session_id, user_text=user_query)

    assert result.session_id == session_id
    assert result.user_text == user_query
    assert "Starter" in result.agent_text or "$499" in result.agent_text
    assert result.lead.name == "Linda"
    assert result.lead.company == "FinTech Solutions"
    assert result.lead.lead_score >= 25
    assert len(result.rag_sources) > 0
    assert result.audio_base64 is not None  # TTS output generated


@pytest.mark.asyncio
async def test_orchestrator_audio_turn():
    """Verify orchestrator accepts audio input, runs STT, and returns speech response."""
    session_id = "orch-test-audio-1"
    mock_audio = b"TEXT: We need to automate lead qualification for our enterprise sales team."

    result = await orchestrator.process_turn(session_id=session_id, audio_bytes=mock_audio)

    assert "lead qualification" in result.user_text
    assert result.lead.need is not None
    assert result.audio_base64 is not None
    assert result.audio_content_type in ["audio/mpeg", "audio/wav"]


@pytest.mark.asyncio
async def test_orchestrator_multi_turn_to_booking():
    """
    Simulate full multi-turn sales interaction:
    Turn 1: Prospect introduces themselves and problem.
    Turn 2: Prospect gives budget and timeline.
    Turn 3: Prospect asks to schedule a demo -> triggers calendar booking & CRM sync.
    """
    session_id = f"orch-multi-{uuid.uuid4().hex[:8]}"

    # Turn 1
    t1 = "Hello, I am Richard Sterling, VP of Operations at Horizon Cloud. We are losing leads after business hours."
    r1 = await orchestrator.process_turn(session_id=session_id, user_text=t1)
    assert r1.lead.name == "Richard Sterling"
    assert r1.lead.company == "Horizon Cloud"
    assert r1.lead.lead_score >= 50

    # Turn 2
    t2 = "We have an approved budget of $12,000 and want to launch in Q1."
    r2 = await orchestrator.process_turn(session_id=session_id, user_text=t2)
    assert r2.lead.budget is not None
    assert r2.lead.timeline is not None
    assert r2.lead.qualification_status == LeadStatus.HOT_LEAD

    # Turn 3: Booking intent
    t3 = "Yes please, let's schedule the demo."
    r3 = await orchestrator.process_turn(session_id=session_id, user_text=t3)
    assert r3.appointment is not None
    assert r3.crm_synced is True
    assert "confirmed your demonstration" in r3.agent_text.lower()


def test_rest_chat_endpoint(client):
    """Verify REST API /api/chat endpoint."""
    payload = {
        "session_id": "rest-sess-1",
        "text": "Hello, does Fortivex AI have SOC2 compliance?"
    }
    resp = client.post("/api/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "SOC2" in data["agent_text"]
    assert len(data["rag_sources"]) > 0


def test_websocket_voice_gateway_text_and_audio(client):
    """Verify full-duplex WebSocket communication with JSON text frames and audio payloads."""
    session_id = "ws-test-session"

    with client.websocket_connect(f"/ws/voice/{session_id}") as websocket:
        # 1. Send JSON text frame
        msg1 = {
            "type": "text",
            "text": "Hi, I am Carlos, Head of Sales at SolarGrid. How does your voice AI compare to human SDRs?",
            "language": "en"
        }
        websocket.send_text(json.dumps(msg1))

        # Receive turn result frame
        resp1_raw = websocket.receive_text()
        resp1 = json.loads(resp1_raw)

        assert resp1["type"] == "turn_result"
        assert resp1["session_id"] == session_id
        assert resp1["lead"]["name"] == "Carlos"
        assert resp1["lead"]["company"] == "SolarGrid"
        assert resp1["audio_base64"] is not None

        # 2. Send mock audio frame (Base64 audio)
        mock_audio_bytes = b"TEXT: What is the cost for the growth tier?"
        audio_b64 = base64.b64encode(mock_audio_bytes).decode("utf-8")
        msg2 = {
            "type": "audio",
            "data": audio_b64
        }
        websocket.send_text(json.dumps(msg2))

        # Receive turn result frame
        resp2_raw = websocket.receive_text()
        resp2 = json.loads(resp2_raw)

        assert resp2["type"] == "turn_result"
        assert "cost" in resp2["user_text"].lower() or "growth" in resp2["user_text"].lower()
        assert resp2["audio_base64"] is not None


def test_websocket_real_microphone_voice_and_transcription(client):
    """
    Verify the bug fix for real microphone voice interaction:
    Ensures that when a user speaks into the microphone (e.g. Rahul from TechNova),
    the WebSocket gateway receives and processes the actual spoken message,
    populates lead profile, extracts authority/role ('Sales Manager'),
    generates contextual AI response, and NEVER defaults to the hardcoded demo string.
    """
    session_id = f"mic-test-{uuid.uuid4().hex[:6]}"

    with client.websocket_connect(f"/ws/voice/{session_id}") as websocket:
        spoken_utterance = "Hi, my name is Rahul. I am the Sales Manager at TechNova. Can you explain what your AI voice agent does?"
        raw_audio_sample = b"\x1aE\xdf\xa3\x00\x00\x00\x00"  # WebM audio container chunk
        msg = {
            "type": "audio",
            "data": base64.b64encode(raw_audio_sample).decode("utf-8"),
            "text": spoken_utterance
        }
        websocket.send_text(json.dumps(msg))

        resp_raw = websocket.receive_text()
        resp = json.loads(resp_raw)

        assert resp["type"] == "turn_result"
        assert resp["session_id"] == session_id
        # 1. Must match actual spoken message
        assert resp["user_text"] == spoken_utterance
        assert "interested in Fortivex AI" not in resp["user_text"]
        # 2. Must extract actual lead details
        assert resp["lead"]["name"] == "Rahul"
        assert resp["lead"]["company"] == "TechNova"
        assert resp["lead"]["authority"] == "Sales Manager"
        assert resp["lead"]["lead_score"] >= 25
        # 3. Must generate voice response
        assert resp["audio_base64"] is not None
        assert resp["agent_text"] is not None
        assert len(resp["agent_text"]) > 10
