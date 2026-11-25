import base64
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.storage.database import get_db
from app.rag.retriever import get_retriever
from app.intelligence.orchestrator import SalesAgentOrchestrator, OrchestratorTurnResult
from app.analytics.insights import BusinessInsightExtractor

# Logger setup
logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger("sales_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing persistence and RAG index."""
    logger.info("Initializing Fortivex AI Autonomous Sales Voice Agent...")
    db = get_db()
    db.init_db()
    retriever = get_retriever()
    logger.info(f"Loaded {len(retriever.vector_store.documents)} knowledge base documents.")
    yield
    logger.info("Shutting down sales agent services.")


app = FastAPI(
    title="Autonomous AI Sales Voice Agent",
    description="Fortivex AI Technical Evaluation - Autonomous B2B Sales Voice Agent with STT/TTS, RAG, BANT Lead Qualification, and CRM Sync.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Central orchestrator singleton
orchestrator = SalesAgentOrchestrator()
analytics_extractor = BusinessInsightExtractor()

# Mount static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root_index():
    """Serves the interactive voice agent evaluation UI."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Autonomous AI Sales Voice Agent API is online. Access /health or /docs."}


class ChatRequest(BaseModel):
    session_id: str
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    language: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Autonomous AI Sales Voice Agent",
        "version": "1.0.0",
        "stt_provider": settings.STT_PROVIDER,
        "tts_provider": settings.TTS_PROVIDER,
        "llm_provider": settings.LLM_PROVIDER
    }


@app.get("/api/session/{session_id}")
async def get_session_state(session_id: str):
    """Retrieves lead profile, conversation turns, and appointment status for a session."""
    db = get_db()
    lead = db.get_lead_by_session(session_id)
    call_log = db.get_call_log_by_session(session_id)
    appointments = db.get_appointments_by_lead(lead.id) if lead and lead.id else []
    crm_records = db.list_crm_records(lead.id) if lead and lead.id else []

    memory = orchestrator.get_or_create_memory(session_id)

    return {
        "session_id": session_id,
        "lead": lead,
        "turns": [turn.model_dump() for turn in memory.turns],
        "appointments": appointments,
        "crm_records": crm_records,
        "call_log": call_log
    }


@app.post("/api/chat", response_model=OrchestratorTurnResult)
async def chat_turn(req: ChatRequest):
    """REST API endpoint for text/audio dialogue turns."""
    audio_bytes = base64.b64decode(req.audio_base64) if req.audio_base64 else None
    result = await orchestrator.process_turn(
        session_id=req.session_id,
        user_text=req.text,
        audio_bytes=audio_bytes,
        language_override=req.language
    )
    return result


@app.post("/api/session/{session_id}/end")
async def end_call_session(session_id: str, duration_seconds: int = 120):
    """
    Finalizes a sales call session, computes business insights and call summary,
    and persists CallLog into the SQLite database.
    """
    db = get_db()
    memory = orchestrator.get_or_create_memory(session_id)
    lead = db.get_lead_by_session(session_id)
    appointments = db.get_appointments_by_lead(lead.id) if lead and lead.id else []
    appointment = appointments[0] if appointments else None

    call_log = await analytics_extractor.generate_call_analytics(
        session_id=session_id,
        turns=memory.turns,
        lead=lead,
        appointment=appointment,
        duration_seconds=duration_seconds
    )
    return call_log


@app.get("/api/session/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Retrieves the CallLog, summary, and sales insights for a session."""
    db = get_db()
    call_log = db.get_call_log_by_session(session_id)
    if not call_log:
        raise HTTPException(status_code=404, detail="Analytics not found for session. Ensure call has been finalized.")
    return call_log


@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Aggregates executive metrics across all stored sales calls."""
    db = get_db()
    leads = db.list_leads()
    appointments = db.list_appointments()
    total_leads = len(leads)
    qualified = sum(1 for l in leads if l.qualification_status.value in ["QUALIFIED", "HOT_LEAD"])
    hot = sum(1 for l in leads if l.qualification_status.value == "HOT_LEAD")
    avg_score = round(sum(l.lead_score for l in leads) / total_leads, 1) if total_leads > 0 else 0

    return {
        "total_prospects": total_leads,
        "qualified_leads": qualified,
        "hot_leads": hot,
        "confirmed_appointments": len(appointments),
        "average_lead_score": avg_score,
        "conversion_pipeline_rate": f"{round((qualified / total_leads * 100), 1)}%" if total_leads > 0 else "0%"
    }


@app.websocket("/ws/voice/{session_id}")
async def websocket_voice_endpoint(websocket: WebSocket, session_id: str):
    """
    Real-Time Full-Duplex Voice WebSocket Gateway.
    Receives customer audio streams (or JSON text/audio payloads)
    and streams back AI spoken responses and real-time sales intelligence.
    """
    await websocket.accept()
    logger.info(f"WebSocket voice session opened: {session_id}")

    try:
        while True:
            # Receive either binary audio data or text/JSON frame
            message = await websocket.receive()

            user_text: Optional[str] = None
            audio_bytes: Optional[bytes] = None
            language: Optional[str] = None

            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "audio" and data.get("data"):
                        audio_bytes = base64.b64decode(data["data"])
                        if data.get("text") and str(data.get("text")).strip():
                            user_text = str(data.get("text")).strip()
                    elif data.get("type") == "text":
                        user_text = str(data.get("text", "")).strip()
                    language = data.get("language")
                except json.JSONDecodeError:
                    # Treat raw text string as user utterance
                    user_text = message["text"].strip()

            # Process turn through central sales agent orchestrator
            turn_result = await orchestrator.process_turn(
                session_id=session_id,
                user_text=user_text,
                audio_bytes=audio_bytes,
                language_override=language
            )

            # Send back response metadata and audio payload
            response_payload = {
                "type": "turn_result",
                "session_id": turn_result.session_id,
                "user_text": turn_result.user_text,
                "agent_text": turn_result.agent_text,
                "detected_language": turn_result.detected_language,
                "sentiment": turn_result.sentiment.model_dump(),
                "lead": turn_result.lead.model_dump(),
                "rag_sources": turn_result.rag_sources,
                "appointment": turn_result.appointment.model_dump() if turn_result.appointment else None,
                "crm_synced": turn_result.crm_synced,
                "audio_base64": turn_result.audio_base64,
                "audio_content_type": turn_result.audio_content_type
            }

            await websocket.send_text(json.dumps(response_payload))

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket voice loop for {session_id}: {str(e)}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
