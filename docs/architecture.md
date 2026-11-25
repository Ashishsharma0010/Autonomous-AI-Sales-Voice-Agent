# System Architecture Documentation

## 1. Overview

The **Fortivex AI Autonomous Sales Voice Agent** is built as a real-time, low-latency, modular system designed to simulate human sales development representative (SDR) conversations. It integrates live bidirectional audio streaming over WebSockets, speech-to-text (STT), multilingual natural language understanding (NLU), retrieval-augmented generation (RAG), BANT lead qualification, calendar booking, CRM synchronisation, post-call analytics, and neural text-to-speech (TTS).

---

## 2. High-Level System Architecture Diagram

```mermaid
flowchart TB
    subgraph Client["Client Tier (Web Browser / Phone Bridge)"]
        UI["Frontend Cockpit (HTML5 / JS / Tailwind CSS)"]
        MIC["Microphone Audio Stream (MediaRecorder / Opus)"]
        SPK["Browser Audio Output (Web Audio API / HTML5 Audio)"]
    end

    subgraph Gateway["API & Gateway Tier (FastAPI / Uvicorn)"]
        WS["WebSocket Voice Gateway (/ws/voice/{session_id})"]
        REST["REST API (/api/chat, /api/session/*, /health)"]
        STATIC["Static Asset Server (/)"]
    end

    subgraph CoreEngine["Core Orchestration & Intelligence Engine"]
        ORCH["SalesAgentOrchestrator (orchestrator.py)"]
        LANG["LanguageDetector (language_detector.py)"]
        SENT["SentimentAnalyzer (sentiment.py)"]
        BANT["LeadQualifier (lead_qualifier.py)"]
        MEM["ConversationMemory & EntityStore (buffer.py, entity_store.py)"]
    end

    subgraph AudioEngine["Voice Pipeline Engine"]
        STT["SpeechToTextService (stt.py)\n[Groq Whisper / OpenAI Whisper / Mock]"]
        TTS["TextToSpeechService (tts.py)\n[Edge Neural TTS / OpenAI TTS / Mock WAV]"]
    end

    subgraph RAGStore["RAG Knowledge Engine"]
        VEC["VectorStore (vector_store.py)\n[TF-IDF + Cosine Similarity Index]"]
        RET["RAGRetriever (retriever.py)"]
        KB[("Knowledge Base Docs\nproducts.json | faqs.json | objections.json")]
    end

    subgraph Integrations["Business Integration Tools"]
        CAL["CalendarTool (calendar_tool.py)\n[Slot Search & Conflict Detection]"]
        CRM["CRMTool (crm_tool.py)\n[Mock / HubSpot / Webhook Sync]"]
    end

    subgraph AnalyticsEngine["Analytics & Insights Engine"]
        SUM["CallSummarizer (summarizer.py)"]
        INS["BusinessInsightExtractor (insights.py)"]
    end

    subgraph DataStore["Persistence Tier (SQLite)"]
        DB[("SQLite Database (sales_agent.db)\nleads | appointments | call_logs | crm_records")]
    end

    %% Audio and Interaction Flows
    MIC -->|Binary Audio Stream (WebM/Opus)| WS
    WS -->|Audio Bytes| ORCH
    ORCH -->|Raw Audio| STT
    STT -->|Transcript Text| ORCH
    ORCH -->|User Utterance| LANG
    ORCH -->|Transcript & Context| SENT
    ORCH -->|Utterance & Extracted Signals| BANT
    ORCH -->|Turn Exchange| MEM
    MEM <-->|State & History Sync| DB

    %% RAG Retrieval
    ORCH -->|Query Keyword Search| RET
    RET <-->|Cosine Similarity Search| VEC
    VEC <-->|Document Ingestion| KB
    RET -->|Grounding Context| ORCH

    %% Booking & CRM
    ORCH -->|Booking Intent Trigger| CAL
    CAL <-->|Conflict Checking| DB
    CAL -->|Confirmed Slot| ORCH
    ORCH -->|Qualified Lead Sync| CRM
    CRM -->|Sync Audit Log| DB

    %% Audio Synthesis
    ORCH -->|Generated Agent Response| TTS
    TTS -->|Synthesized Audio (MP3/WAV)| ORCH
    ORCH -->|JSON Turn Result + Base64 Audio| WS
    WS -->|Audio & Metadata Payload| UI
    UI --> SPK

    %% REST & Analytics
    UI -->|End Call Request| REST
    REST -->|Session Finalisation| SUM
    REST -->|Turn Log & Lead Profile| INS
    SUM -->|Executive Summary| INS
    INS -->|Persist CallLog| DB
    REST -->|Fetch Overview & Session Analytics| DB
```

---

## 3. Subsystem Breakdown

### 3.1. Client Tier (Frontend Cockpit)
- **Role:** Provides real-time operator monitoring and interactive sales testing.
- **Microphone Pipeline:** Captures live audio using the browser's `navigator.mediaDevices.getUserMedia()` and `MediaRecorder` API encoded into WebM/Opus.
- **WebSocket Client:** Streams raw binary packets directly over `ws://127.0.0.1:8000/ws/voice/{session_id}`.
- **Audio Rendering:** Receives base64-encoded audio responses (MP3 or RIFF WAV) and renders sound seamlessly through an HTML5 Audio element.
- **Visual Telemetry:** Renders real-time BANT scorecard updates, lead-score progress bars, sentiment indicator, appointment details, and post-call analytics modal.

### 3.2. Gateway Tier (FastAPI)
- **Role:** Handles protocol translation, session routing, and connection lifecycles.
- **WebSocket Protocol:** Supports both binary stream frames and formatted JSON envelopes (`type: "text"` or `type: "audio"`).
- **REST Endpoints:** Exposes health status, session inspection, dialogue chat endpoints, and analytics summaries.
- **Static Assets:** Directly mounts the single-page application from `app/static/` with zero build-step overhead.

### 3.3. Core Orchestrator & Intelligence
- **Role:** Coordinates sequential multi-model execution per turn (`app/intelligence/orchestrator.py`).
- **Language Detection:** Analyzes utterance lexical features across English, Spanish, Hindi, French, and German.
- **Sentiment Analysis:** Computes conversational sentiment polarity and classifies explicit objection themes (budget, competition, timing, product fit, security).
- **BANT Lead Qualifier:** Continuously extracts Budget, Authority, Need, and Timeline parameters using regex heuristics, evaluating numerical scores (0–100) and status transitions (`UNQUALIFIED`, `QUALIFYING`, `QUALIFIED`, `HOT_LEAD`).
- **Contextual Memory:** Maintains a 15-turn sliding dialogue window backed by entity extraction and persistent SQLite records.

### 3.4. RAG Knowledge Engine
- **Role:** Grounds responses in verified product literature without hallucinations.
- **TF-IDF Vector Store:** Pure Python term-frequency inverse-document-frequency representation with cosine similarity.
- **Knowledge Base:** Curated JSON files covering product tiers, frequently asked questions, and tactical objection handling.
- **Attribution Guardrails:** Responses citing performance statistics (e.g., cost savings or language counts) are explicitly framed as catalog knowledge rather than independent claims.

### 3.5. Calendar & CRM Integrations
- **Calendar Engine:** Checks existing bookings in SQLite, scans dynamic forward dates (today + 1 to today + 20), selects valid daytime slots, and prevents conflicting reservations.
- **CRM Engine:** Supports internal database storage, external webhook event dispatching, and HubSpot-ready payload structuring.

### 3.6. Voice Pipeline
- **STT Service:** Cascading support prioritizing Groq Whisper API (`whisper-large-v3-turbo`), OpenAI Whisper, and local deterministic fallback with automatic audio container magic-byte identification.
- **TTS Service:** Zero-cost Microsoft Edge Neural TTS generating natural speech (`en-US-AriaNeural`, `es-ES-ElviraNeural`, `hi-IN-SwaraNeural`, etc.), falling back to OpenAI TTS and standard RIFF WAV generation.

### 3.7. Analytics & Persistence
- **Call Summariser:** Synthesises conversation turns into a cohesive narrative summary, key takeaways, and action items.
- **Business Insight Extractor:** Calculates conversion likelihood, audit trails, and commercial recommendations.
- **Database:** SQLite relational storage using `aiosqlite` with normalized tables for leads, appointments, call logs, and CRM dispatches.
