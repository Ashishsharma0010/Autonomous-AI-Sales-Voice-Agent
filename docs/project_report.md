# One-Page Project Report: Autonomous AI Sales Voice Agent

**Project:** Fortivex AI Technical Assessment — Autonomous B2B Sales Voice Agent  
**Agent Persona:** Aria (Inbound & Outbound Sales Specialist)  
**Author:** AI Technical Assessment Candidate  
**Date:** November 2026  
**Status:** Complete & Verified (46/46 Tests Passing)  

---

### 1. Executive Summary & Business Problem
B2B software companies face significant lead-funnel leakage when inbound prospects wait for manual callbacks, when human Sales Development Representatives (SDRs) are unavailable outside business hours, or when qualification criteria are applied inconsistently across high inquiry volumes.

This project delivers **Aria**, an autonomous, real-time AI sales voice agent designed to handle inbound and outbound voice discovery calls, qualify prospects using the **BANT** framework, answer product and commercial queries via **RAG**, schedule discovery meetings directly onto calendar systems, and synchronize structured records to CRM pipelines.

---

### 2. Proposed Solution & Core Features
The solution is an end-to-end autonomous voice platform built with Python 3.14, FastAPI, and modular services:
- **Zero-Latency Voice Gateway:** Full-duplex WebSocket audio streaming supporting live browser microphone input via WebM/Opus and instant playback.
- **Real-Time Speech Processing:** Multi-tier STT (Groq/OpenAI Whisper with deterministic mock) and high-fidelity neural voice synthesis (Microsoft Edge Neural TTS without external API billing).
- **RAG Knowledge Base:** Built-in TF-IDF cosine-similarity retriever over curated product specifications, pricing models, objection playbooks, and FAQs.
- **Intelligent BANT Qualification:** Dynamic extraction of Budget, Authority, Need, and Timeline parameters with real-time lead scoring (0–100 scale) and lifecycle state transitions.
- **Automated Scheduling & CRM Sync:** Conflict-aware appointment reservation across dynamic forward date slots and automated CRM synchronization.
- **Post-Call Analytics & Insights:** Automated call summarisation, sentiment trajectory tracking, objection cataloging, and deal conversion probability scoring.

---

### 3. Technical Implementation Overview

| Architectural Layer | Implementation Detail | Operational Benefit |
| :--- | :--- | :--- |
| **Backend & WebSockets** | FastAPI, Uvicorn, AsyncIO, WebSockets | Non-blocking, asynchronous handling of concurrent calls |
| **Persistence** | SQLite relational database (`aiosqlite`, Pydantic v2) | Lightweight, zero-config local storage for leads & logs |
| **Knowledge Retrieval** | TF-IDF Vector Space, Cosine Similarity, JSON storage | Deterministic, offline-capable RAG with zero latency spikes |
| **Voice Processing** | Microsoft Edge Neural TTS + Whisper STT Engine | Realistic natural speech synthesis with zero per-minute API cost |
| **Operator Cockpit** | HTML5, Tailwind CSS, MediaRecorder & Web Audio APIs | Comprehensive single-page interface for live operator monitoring |

---

### 4. Business Value
- **Immediate Engagement:** Answers inbound voice and text inquiries automatically at any time without hold queues.
- **Consistent Qualification Quality:** Applies objective BANT scoring rules (Budget, Authority, Need, Timeline plus sentiment modifiers) across every interaction.
- **Automated Documentation & CRM Hygiene:** Automatically creates lead records, books conflict-free calendar slots, logs CRM sync status, and generates post-call executive summaries and action items.
- **Knowledge-Grounded Responses:** Grounds pricing, security, and objection responses in curated product documentation rather than unverified claims.

---

### 5. Scalability & Commercial Feasibility
- **Current State:** Self-contained local deployment with 46 automated unit, integration, and end-to-end tests verifying all implementation phases. Supports live Groq Whisper STT + Groq LLM (`qwen/qwen3.8-27b`) as well as offline/mock fallbacks without required API keys.
- **Path to 100,000 Users:** Modular separation allows straightforward horizontal scaling: migrating SQLite to managed PostgreSQL, deploying FastAPI instances behind a load balancer via Kubernetes, introducing Redis for distributed session memory, and scaling dedicated STT/TTS inference workers (see [scalability.md](scalability.md)).
- **Commercial Feasibility:** Using free-tier-capable Edge Neural TTS for speech synthesis and local TF-IDF retrieval for RAG minimizes per-call external service dependencies while keeping commercial LLM/STT providers configurable via `.env`.
