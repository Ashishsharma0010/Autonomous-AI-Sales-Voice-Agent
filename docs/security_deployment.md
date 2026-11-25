# Security, Configuration, and Deployment Documentation

---

## 1. Implemented Security Controls

The current codebase implements the following security and data governance practices:

### 1.1. Data Validation & Boundary Sanitisation
- **Strict Pydantic v2 Models:** All API payloads (`ChatRequest`, `Lead`, `Appointment`, `CallLog`, `CRMRecord`) enforce explicit typing and schema validation.
- **Audio Magic-Byte Validation:** The speech-to-text service (`app/audio/stt.py`) inspects raw binary magic bytes to ensure incoming audio strictly matches legitimate multimedia containers (WebM, RIFF WAV, OGG, MP3, M4A), rejecting malicious binary uploads.
- **Database Parameterisation:** SQLite operations in `app/storage/database.py` use strictly parameterised SQL queries (`?` placeholders), preventing SQL injection vulnerabilities.

### 1.2. Error Isolation & Graceful Degradation
- **WebSocket Exception Boundaries:** Unhandled client disconnections and bad frames are trapped in structured `try/except` blocks in `app/main.py`, sending clean JSON error frames before orderly socket cleanup.
- **Fallback Cascades:** The system automatically falls back gracefully across all model providers:
  - Groq Whisper → OpenAI Whisper → Deterministic STT.
  - Groq Llama 3.3 → OpenAI GPT → Deterministic RAG Response.
  - Edge Neural TTS → OpenAI TTS → Standard RIFF WAV Fallback.
  - External CRM Sync → Local DB Audit Log with `FAILED` or `SYNCED` flags.

---

## 2. Environment Variables & Configuration

The application is configured using Pydantic Settings (`app/config.py`), reading from environment variables or a `.env` file located in the project root.

| Environment Variable | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `HOST` | string | `0.0.0.0` | Network binding interface for FastAPI |
| `PORT` | integer | `8000` | Port for the Uvicorn web server |
| `DEBUG` | boolean | `True` | Enables debug logging and verbose tracebacks |
| `ENVIRONMENT` | string | `development` | Deployment environment flag (`development`, `production`) |
| `DATABASE_PATH` | string | `sales_agent.db` | Path to the local SQLite database file |
| `LLM_PROVIDER` | string | `groq` | Active LLM provider (`groq`, `openai`, `gemini`) |
| `LLM_MODEL` | string | `llama-3.3-70b-versatile` | LLM model identifier |
| `GROQ_API_KEY` | string | `""` | Optional API key for Groq Cloud |
| `OPENAI_API_KEY` | string | `""` | Optional API key for OpenAI services |
| `GEMINI_API_KEY` | string | `""` | Optional API key for Google Gemini |
| `STT_PROVIDER` | string | `groq` | Active STT provider (`groq`, `openai`, `browser`) |
| `STT_MODEL` | string | `whisper-large-v3-turbo` | Whisper STT model variant |
| `TTS_PROVIDER` | string | `edge` | Active TTS provider (`edge`, `openai`, `elevenlabs`) |
| `TTS_VOICE` | string | `en-US-AriaNeural` | Default voice identifier for Edge Neural TTS |
| `CALENDAR_MODE` | string | `mock` | Calendar integration mode (`mock`, `google`) |
| `CRM_MODE` | string | `mock` | CRM integration mode (`mock`, `hubspot`, `webhook`) |
| `CRM_WEBHOOK_URL`| string | `""` | Webhook URL for external CRM dispatches |

---

## 3. Deployment Instructions

### 3.1. Local Development Execution
```powershell
# 1. Navigate to repository root
cd c:\Users\HP\OneDrive\Desktop\Fortivex_Ai_Assesment

# 2. Activate virtual environment (if applicable) and install dependencies
pip install -r requirements.txt

# 3. Initialize configuration
copy .env.example .env

# 4. Start the application
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3.2. Production Container Deployment (Docker)

To deploy the service in a containerised production environment, use the standard Dockerfile pattern below:

```dockerfile
# Multi-stage lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install system audio dependencies for TTS/STT
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/
COPY pytest.ini .
COPY .env.example .env

EXPOSE 8000

# Run with production Uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and execute container:
```bash
docker build -t fortivex-sales-voice-agent:v1 .
docker run -d -p 8000:8000 --name sales-agent fortivex-sales-voice-agent:v1
```

---

## 4. Known Implementation Limitations

1. **Database Concurrency:** The local SQLite storage engine is optimized for single-node development. Concurrent high-frequency writes under load require migration to PostgreSQL.
2. **In-Memory Session Affinity:** Dialogue sessions are maintained in an in-memory dictionary. Multi-process clustering requires external session caching via Redis.
3. **Browser Audio Permissions:** Web browser microphone streaming requires HTTPS or `localhost` / `127.0.0.1` origins due to browser security policies.
4. **Lexical Language Detection:** Language classification relies on heuristic keyword token scanning, which is effective for standard phrases but less sensitive for single-word non-English utterances.
