**Agent Name:** Aria  
**Project:** Autonomous AI Sales Voice Agent  
**Category:** Generative AI · Agentic AI · Voice Automation  
**Backend:** Python 3.14 · FastAPI · WebSockets  
**AI Capabilities:** LLM Integration · RAG · STT · TTS · BANT Lead Qualification · Sentiment Analysis  
**Storage:** SQLite · aiosqlite  
**Testing:** 44 automated tests passing

Overview

Aria is an autonomous AI-powered sales voice agent built to automate B2B sales conversations. It interacts with potential customers through voice and text, understands customer requirements, qualifies leads using the BANT framework, retrieves relevant business information using RAG, and automates appointment booking and CRM synchronization.

The platform also provides real-time sentiment analysis, contextual conversation memory, lead scoring, and post-call business intelligence

---

## Table of Contents

1. [Business Problem & Objectives](#1-business-problem--objectives)
2. [Implemented Features](#2-implemented-features)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Complete Conversation Workflow](#5-complete-conversation-workflow)
6. [Environment Setup](#6-environment-setup)
7. [Running the Application](#7-running-the-application)
8. [API Reference](#8-api-reference)
9. [WebSocket Protocol](#9-websocket-protocol)
10. [Knowledge Base](#10-knowledge-base)
11. [Project Structure](#11-project-structure)
12. [Testing](#12-testing)
13. [Scalability Notes](#13-scalability-notes)
14. [Known Limitations](#14-known-limitations)

---

## 1. Business Problem & Objectives

B2B software companies lose significant revenue because human Sales Development Representatives (SDRs) cannot respond to every inbound enquiry instantly, work around the clock, or maintain consistent qualification quality across hundreds of simultaneous conversations.

**This project implements an Autonomous AI Sales Voice Agent** that:

- Answers inbound sales enquiries in real time via voice.
- Qualifies prospects using the BANT framework (Budget, Authority, Need, Timeline).
- Retrieves accurate product and pricing information from a curated knowledge base.
- Books discovery-call appointments automatically.
- Synchronises qualified lead data to a CRM system.
- Generates a post-call summary and business insights after every conversation.

---

## 2. Implemented Features

### Phase 1 — Storage, Configuration & Data Models
- Pydantic v2 data models: `Lead`, `Appointment`, `CallLog`, `CRMRecord`, `CallTurn`
- SQLite persistence via `aiosqlite` with a `DatabaseManager` singleton
- Centralised `Settings` class with `.env` and environment-variable loading
- Support for toggling between providers without code changes

### Phase 2 — RAG Knowledge Base
- JSON knowledge-base documents: products & pricing, objection handling, FAQs
- TF-IDF cosine-similarity vector store (no external embedding service required)
- `RAGRetriever` returns the top-*k* relevant passages for any query
- Retrieved context is injected into the LLM system prompt

### Phase 3 — Intelligence, Sentiment & Lead Qualification
- **Contextual Memory:** 15-turn sliding-window conversation buffer, persisted to the database
- **Sentiment Analysis:** lexical scoring into five categories — POSITIVE, NEUTRAL, HESITANT, NEGATIVE, FRUSTRATED; detects seven objection types
- **BANT Qualifier:** regex-based extraction of Budget, Authority, Need, and Timeline signals from free-form speech
- **Lead Scoring:** 25 points per BANT dimension (max 100); sentiment modifier ±10/15; thresholds — HOT_LEAD ≥ 75, QUALIFIED ≥ 50, QUALIFYING ≥ 25

### Phase 4 — Calendar & CRM Integration
- **CalendarTool:** Dynamic slot search (today + 1 to today + 20 days, business hours); conflict checking against the database; generates a meeting link
- **CRMTool:** Supports Mock CRM (local DB), Webhook mode, and HubSpot mode; records failed syncs with a FAILED status

### Phase 5 — Voice Pipeline (STT & TTS)
- **SpeechToTextService:** Groq Whisper → OpenAI Whisper → deterministic fallback; magic-bytes detection for WebM, WAV, OGG, MP3, M4A
- **TextToSpeechService:** Microsoft Edge Neural TTS (no API key required) → OpenAI TTS → mock WAV fallback
- **LanguageDetector:** Detects en, es, hi, fr, de; maps to Edge neural voices per language

### Phase 6 — Orchestration & WebSocket Gateway
- **SalesAgentOrchestrator:** Central coordinator running the full pipeline each turn — STT → language detection → sentiment → BANT evaluation → DB persistence → RAG retrieval → booking intent detection → LLM response generation → TTS synthesis
- **LLM cascade:** Groq (llama-3.3-70b-versatile) → OpenAI → deterministic rule-based fallback
- **WebSocket gateway** at `/ws/voice/{session_id}`: accepts binary audio frames or JSON `{type: "audio"/"text"}` payloads; returns `turn_result` JSON including base64-encoded audio
- **REST API** at `/api/chat` for text-only interactions

### Phase 7 — Call Summarisation & Business Insights
- **CallSummarizer:** Generates a prose summary, key takeaways list, and action items list from the conversation turns
- **BusinessInsightExtractor:** Produces sentiment trajectory, objection catalogue, conversion-probability score, BANT audit, and commercial recommendation; persists a `CallLog` to the database
- **API endpoint** to retrieve per-session analytics and an executive overview endpoint

### Phase 8 — Frontend Cockpit & End-to-End Verification
- Single-page web UI served at `/` (no build step required)
- Push-to-talk microphone (MediaRecorder API, WebM/Opus → binary WebSocket)
- Real-time conversation transcript panel
- Live Lead Intelligence panel: BANT scorecard, lead-score progress bar, status badge
- Turn Signals panel: sentiment indicator, detected language, appointment card, CRM sync status
- Post-Call Executive Summary modal triggered by the "End Call & View Summary" button
- Full 9-step automated end-to-end test (`tests/test_end_to_end.py`)

---

## 3. Technology Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| API Server | FastAPI 0.115 | async, OpenAPI auto-docs at `/docs` |
| WebSocket | FastAPI WebSocket | Full-duplex voice gateway |
| Data Models | Pydantic v2 | Strict validation, serialisation |
| Database | SQLite + aiosqlite | Zero-dependency persistence |
| RAG | TF-IDF cosine similarity | No external embedding service |
| LLM | Groq / OpenAI / Rule-based | Cascading fallback chain |
| STT | Groq Whisper / OpenAI Whisper | Deterministic mock for tests |
| TTS | Microsoft Edge Neural TTS | No API key required; `edge-tts` package |
| Language Detection | Custom lexical detector | en, es, hi, fr, de |
| Frontend | Vanilla HTML / CSS / JS | MediaRecorder API, WebSocket |
| Testing | pytest + pytest-asyncio | 44 tests, strict asyncio mode |
| Config | pydantic-settings + python-dotenv | `.env` file or environment variables |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Browser / Client                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Microphone  │  │  Text Input  │  │  Frontend Cockpit UI  │  │
│  │ MediaRecorder│  │  / Prompts   │  │  (index.html)         │  │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬───────────┘  │
│         │  binary audio  │  JSON text             │ WebSocket    │
└─────────┼────────────────┼────────────────────────┼─────────────┘
          │                │                        │
          ▼                ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application (app/main.py)             │
│  GET /          POST /api/chat       WS /ws/voice/{session_id}  │
│  GET /health    GET /api/session/…   POST /api/session/…/end    │
│                 GET /api/analytics/… GET /api/session/…/analytics│
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              SalesAgentOrchestrator (orchestrator.py)            │
│                                                                  │
│  Audio → STT → Language Detection → Sentiment Analysis          │
│       → BANT Qualification → DB Persist → RAG Retrieval         │
│       → Booking Intent → LLM Response → Memory Update → TTS     │
└──┬─────────┬────────────┬─────────────┬────────────┬────────────┘
   │         │            │             │            │
   ▼         ▼            ▼             ▼            ▼
┌──────┐ ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐
│ STT  │ │ TTS  │  │   RAG    │  │  Memory  │  │  Calendar /  │
│ stt  │ │ tts  │  │retriever │  │  buffer  │  │  CRM Tools   │
│ .py  │ │ .py  │  │ .py      │  │  .py     │  │  tools/      │
└──┬───┘ └──┬───┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘
   │        │           │             │                │
   ▼        ▼           ▼             ▼                ▼
┌───────┐ ┌──────────┐ ┌─────────┐ ┌────────────┐ ┌──────────┐
│Groq / │ │Edge-TTS /│ │ TF-IDF  │ │ Entity     │ │ SQLite   │
│OpenAI │ │OpenAI TTS│ │ Vector  │ │ Store +    │ │ Database │
│Whisper│ │/ Mock WAV│ │ Store   │ │ Sliding    │ │(database │
└───────┘ └──────────┘ └────┬────┘ │ Window     │ │  .py)    │
                             │      └────────────┘ └──────────┘
                        ┌────┴────┐
                        │  JSON   │
                        │Knowledge│
                        │  Base   │
                        │products │
                        │faqs     │
                        │objections│
                        └─────────┘
```

---

## 5. Complete Conversation Workflow

```
Customer speaks or types
         │
         ▼
1. Audio received via WebSocket (binary) or REST API (base64/text)
         │
         ▼
2. STT — Groq Whisper / OpenAI Whisper / mock
   Audio bytes → transcript text + detected language
         │
         ▼
3. Sentiment Analysis
   Lexical scoring → POSITIVE / NEUTRAL / HESITANT / NEGATIVE / FRUSTRATED
   Objection type detection (budget, competition, timing, quality, security…)
         │
         ▼
4. BANT Signal Extraction
   Regex patterns extract Budget, Authority, Need, Timeline signals
   Lead score updated (25 pts per dimension + sentiment modifier)
   Qualification status updated: UNQUALIFIED → QUALIFYING → QUALIFIED → HOT_LEAD
         │
         ▼
5. Persistence
   Lead profile upserted to SQLite
   Conversation turn appended to memory buffer (15-turn sliding window)
         │
         ▼
6. RAG Retrieval
   User query matched against TF-IDF index (products, FAQs, objections)
   Top-3 passages retrieved and formatted for LLM context
         │
         ▼
7. Booking Intent Detection
   "demo", "meeting", "schedule", "book" keywords trigger CalendarTool
   First conflict-free slot booked; CRMTool syncs lead record
         │
         ▼
8. LLM Response Generation
   System prompt (Aria persona) + conversation history + RAG context → response
   Cascade: Groq → OpenAI → deterministic rule-based fallback
         │
         ▼
9. TTS Synthesis
   Response text → Edge Neural TTS → audio bytes (MP3/WAV)
   Audio encoded as base64 and returned in WebSocket payload
         │
         ▼
10. Frontend receives turn_result JSON
    Transcript updated · Lead panel refreshed · Audio played in browser
         │
         ▼
11. Call End (customer or timeout)
    POST /api/session/{id}/end
    BusinessInsightExtractor generates summary, insights, conversion probability
    CallLog persisted to database
    Post-Call Executive Summary modal displayed in UI
```

---

## 6. Environment Setup

### Prerequisites

- Python 3.11 or higher (tested on Python 3.14)
- pip

### Installation

```powershell
# Clone or extract the project
cd "c:\Users\HP\OneDrive\Desktop\Fortivex_Ai_Assesment"

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and edit as needed:

```powershell
copy .env.example .env
```

**`.env` reference:**

```ini
# ── LLM ──────────────────────────────────────────────────
LLM_PROVIDER=groq           # groq | openai | gemini
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=               # optional — enables real Groq Whisper STT + LLM
OPENAI_API_KEY=             # optional — enables OpenAI Whisper + GPT + OpenAI TTS
GEMINI_API_KEY=             # optional

# ── STT ──────────────────────────────────────────────────
STT_PROVIDER=groq           # groq | openai | browser
STT_MODEL=whisper-large-v3-turbo

# ── TTS ──────────────────────────────────────────────────
TTS_PROVIDER=edge           # edge (default, no key) | openai | elevenlabs
TTS_VOICE=en-US-AriaNeural
ELEVENLABS_API_KEY=         # optional
ELEVENLABS_VOICE_ID=        # optional

# ── Database ─────────────────────────────────────────────
DATABASE_PATH=sales_agent.db

# ── Integrations ─────────────────────────────────────────
CALENDAR_MODE=mock          # mock | google
CRM_MODE=mock               # mock | hubspot | webhook
CRM_WEBHOOK_URL=            # required if CRM_MODE=webhook
```

> **Note:** The application runs fully without any API keys.  
> Edge-TTS works offline. The LLM and STT fall back to deterministic responses automatically.

---

## 7. Running the Application

### Start the Server

```powershell
cd "c:\Users\HP\OneDrive\Desktop\Fortivex_Ai_Assesment"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Open the Frontend Cockpit

Navigate to **http://127.0.0.1:8000** in your browser.

### Interactive API Documentation

Navigate to **http://127.0.0.1:8000/docs** for the auto-generated OpenAPI (Swagger) UI.

### Run the Full Test Suite

```powershell
python -m pytest tests/ -v
```

Expected output:
```
44 passed, 1 warning in ~28s
```

---

## 8. API Reference

### Health Check

```
GET /health
```
Returns service status and active provider configuration.

---

### Chat Turn (REST)

```
POST /api/chat
Content-Type: application/json

{
  "session_id": "string",
  "text": "optional customer utterance",
  "audio_base64": "optional base64-encoded audio bytes",
  "language": "optional override e.g. en, es, hi"
}
```

**Response:** `OrchestratorTurnResult`

```json
{
  "session_id": "...",
  "user_text": "...",
  "agent_text": "...",
  "detected_language": "en",
  "sentiment": { "type": "POSITIVE", "score": 0.6, "objection_type": null },
  "lead": { "name": "...", "company": "...", "lead_score": 75, "qualification_status": "HOT_LEAD", ... },
  "rag_sources": ["products", "faqs"],
  "appointment": { "slot_time": "...", "meeting_link": "...", "status": "CONFIRMED" },
  "crm_synced": true,
  "audio_base64": "...",
  "audio_content_type": "audio/mpeg"
}
```

---

### Session State

```
GET /api/session/{session_id}
```
Returns the full lead profile, conversation turns, appointments, CRM records, and call log for a session.

---

### End Call Session

```
POST /api/session/{session_id}/end?duration_seconds=120
```
Finalises the session. Generates and persists call summary, business insights, and analytics.

---

### Session Analytics

```
GET /api/session/{session_id}/analytics
```
Returns the persisted `CallLog` with summary, key takeaways, action items, conversion probability, and BANT audit.

---

### Analytics Overview

```
GET /api/analytics/overview
```
Aggregates metrics across all calls: total prospects, qualified leads, hot leads, confirmed appointments, average lead score, conversion pipeline rate.

---

## 9. WebSocket Protocol

**Endpoint:** `WS /ws/voice/{session_id}`

### Sending Input

**Option A — Binary audio frame:**
```
[raw audio bytes: WebM/Opus, WAV, OGG, MP3, or M4A]
```

**Option B — JSON text frame:**
```json
{ "type": "text", "text": "customer utterance", "language": "en" }
```

**Option C — JSON audio frame:**
```json
{ "type": "audio", "data": "<base64-encoded audio bytes>", "language": "en" }
```

### Receiving Responses

```json
{
  "type": "turn_result",
  "session_id": "...",
  "user_text": "...",
  "agent_text": "...",
  "detected_language": "en",
  "sentiment": { "type": "POSITIVE", "score": 0.6, "objection_type": null },
  "lead": { ... },
  "rag_sources": [...],
  "appointment": { ... } | null,
  "crm_synced": true | false,
  "audio_base64": "<base64 MP3/WAV>",
  "audio_content_type": "audio/mpeg"
}
```

**Error frame:**
```json
{ "type": "error", "message": "description" }
```

The browser frontend decodes `audio_base64` and plays it using the Web Audio API automatically.

---

## 10. Knowledge Base

The RAG knowledge base lives in `app/rag/data/` as JSON files:

| File | Contents |
|------|----------|
| `products.json` | Starter, Growth, and Enterprise tier descriptions, feature lists, and pricing |
| `faqs.json` | Common questions: integrations, multilingual support, security, ROI |
| `objections.json` | Objection-handling scripts for cost, competition, timing, and trust objections |

> **Important:** All claims in these documents (e.g. cost-reduction figures, language counts) are sourced from the product knowledge base and are presented by the agent as such — not as independently verified facts.

To add new knowledge, append JSON objects to the appropriate file following the existing schema:
```json
{ "id": "unique-id", "title": "Document Title", "content": "Full text content." }
```

---

## 11. Project Structure

```
Fortivex_Ai_Assesment/
├── app/
│   ├── __init__.py
│   ├── config.py                   # Settings & provider config
│   ├── main.py                     # FastAPI app, REST API, WebSocket gateway
│   ├── analytics/
│   │   ├── insights.py             # BusinessInsightExtractor (Phase 7)
│   │   └── summarizer.py           # CallSummarizer (Phase 7)
│   ├── audio/
│   │   ├── language_detector.py    # Language detection & voice mapping (Phase 5)
│   │   ├── stt.py                  # SpeechToTextService (Phase 5)
│   │   └── tts.py                  # TextToSpeechService (Phase 5)
│   ├── intelligence/
│   │   ├── lead_qualifier.py       # BANT extraction & lead scoring (Phase 3)
│   │   ├── orchestrator.py         # Central pipeline coordinator (Phase 6)
│   │   ├── prompts.py              # Aria system prompt (Phase 3)
│   │   └── sentiment.py            # SentimentAnalyzer (Phase 3)
│   ├── memory/
│   │   ├── buffer.py               # ConversationMemory sliding window (Phase 3)
│   │   └── entity_store.py         # SessionEntityStore / BANT fields (Phase 3)
│   ├── rag/
│   │   ├── data/
│   │   │   ├── faqs.json
│   │   │   ├── objections.json
│   │   │   └── products.json
│   │   ├── retriever.py            # RAGRetriever singleton (Phase 2)
│   │   └── vector_store.py         # TF-IDF VectorStore (Phase 2)
│   ├── static/
│   │   └── index.html              # Frontend cockpit single-page UI (Phase 8)
│   ├── storage/
│   │   ├── database.py             # DatabaseManager / SQLite (Phase 1)
│   │   └── models.py               # Pydantic data models (Phase 1)
│   └── tools/
│       ├── calendar_tool.py        # CalendarTool (Phase 4)
│       └── crm_tool.py             # CRMTool (Phase 4)
├── tests/
│   ├── test_analytics.py           # Phase 7 — 4 tests
│   ├── test_audio.py               # Phase 5 — 7 tests
│   ├── test_end_to_end.py          # Phase 8 — 1 test (9-step workflow)
│   ├── test_intelligence.py        # Phase 3 — 8 tests
│   ├── test_orchestrator.py        # Phase 6 — 6 tests
│   ├── test_rag.py                 # Phase 2 — 7 tests
│   ├── test_storage.py             # Phase 1 — 6 tests
│   └── test_tools.py               # Phase 4 — 6 tests
├── docs/
│   ├── architecture.md             # Architecture diagram (Mermaid)
│   ├── project_report.md           # One-page executive report
│   ├── scalability.md              # Scaling to 100,000 users
│   └── security_deployment.md      # Security & deployment guide
├── .env.example                    # Environment variable template
├── pytest.ini                      # pytest configuration
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## 12. Testing

The test suite covers all eight implementation phases with 44 automated tests:

| Test File | Phase | Tests | Coverage |
|-----------|-------|-------|----------|
| `test_storage.py` | 1 | 6 | Settings, DB init, lead lifecycle, appointments, call logs, CRM records |
| `test_rag.py` | 2 | 7 | Knowledge base loading, pricing/objection/FAQ retrieval, multilingual, edge cases |
| `test_intelligence.py` | 3 | 8 | Sliding window memory, DB sync, sentiment, BANT progression, scoring thresholds |
| `test_tools.py` | 4 | 6 | Appointment booking, slot conflicts, invalid ranges, CRM sync, failure handling |
| `test_audio.py` | 5 | 7 | Language detection, voice selection, STT, Edge TTS, mock TTS, container detection |
| `test_orchestrator.py` | 6 | 6 | Health endpoint, text/audio turns, multi-turn to booking, REST chat, WebSocket |
| `test_analytics.py` | 7 | 4 | Summariser, empty dialogue, insight extraction, analytics API |
| `test_end_to_end.py` | 8 | 1 | Complete 9-step business workflow |

Run all tests:
```powershell
python -m pytest tests/ -v
```

---

## 13. Scalability Notes

The current implementation is a single-process, single-instance application suitable for development and demonstration. See [`docs/scalability.md`](docs/scalability.md) for a discussion of the architectural path to production scale.

Key considerations for production:
- Replace SQLite with PostgreSQL and use a connection pool
- Run multiple Uvicorn workers behind a reverse proxy (Nginx)
- Use a message queue (Redis Streams / Kafka) for async audio processing
- Replace the in-memory session dict with a distributed cache (Redis)
- Deploy as containers (Docker) on a Kubernetes cluster or serverless platform

---

## 14. Known Limitations

| Area | Limitation |
|------|-----------|
| STT | Without a Groq or OpenAI API key, speech recognition falls back to a deterministic mock (text prefixed with `TEXT:` from tests); real Whisper transcription requires an API key |
| LLM | Without a Groq or OpenAI API key, responses come from a rule-based deterministic fallback that handles common patterns; nuanced conversation requires an LLM API key |
| Calendar | Only the Mock calendar provider is fully functional without external credentials; Google Calendar integration requires OAuth configuration |
| CRM | Only the Mock CRM provider is fully functional; HubSpot and Webhook modes require external configuration |
| Concurrency | The in-memory session store is not shared across processes; horizontal scaling requires a distributed session backend |
| Language | Language detection is lexical (keyword-based); low-accuracy for short or ambiguous inputs |
| Database | SQLite does not support high write concurrency; production deployments should migrate to PostgreSQL |

---

*Built for the Fortivex AI B2B Technical Assessment — November 2026*
