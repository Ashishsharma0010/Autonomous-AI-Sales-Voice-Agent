import io
from typing import Optional
from pydantic import BaseModel
import httpx
from app.config import settings
from app.audio.language_detector import LanguageDetector


class TranscriptionResult(BaseModel):
    text: str
    language: str
    confidence: float
    provider: str


class SpeechToTextService:
    """
    Speech-to-Text (STT) Service.
    Supports Groq Whisper API (sub-250ms), OpenAI Whisper API, and Mock/Browser fallback.
    Automatically detects container formats (WebM, WAV, OGG, MP3) and spoken language.
    """

    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self._groq_api_key = settings.GROQ_API_KEY
        self._openai_api_key = settings.OPENAI_API_KEY

    @property
    def groq_api_key(self) -> str:
        return self._groq_api_key or settings.GROQ_API_KEY

    @groq_api_key.setter
    def groq_api_key(self, value: str):
        self._groq_api_key = value

    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key or settings.OPENAI_API_KEY

    @openai_api_key.setter
    def openai_api_key(self, value: str):
        self._openai_api_key = value

    @staticmethod
    def detect_audio_container(audio_bytes: bytes) -> tuple[str, str]:
        """Detects audio container from binary magic bytes, returning (filename, mime_type)."""
        if audio_bytes.startswith(b"\x1aE\xdf\xa3"):
            return "recording.webm", "audio/webm"
        elif audio_bytes.startswith(b"RIFF"):
            return "recording.wav", "audio/wav"
        elif audio_bytes.startswith(b"OggS"):
            return "recording.ogg", "audio/ogg"
        elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            return "recording.mp3", "audio/mpeg"
        elif audio_bytes.startswith(b"ftyp") or (len(audio_bytes) > 8 and audio_bytes[4:8] == b"ftyp"):
            return "recording.m4a", "audio/mp4"
        return "recording.wav", "audio/wav"

    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
        language: Optional[str] = None,
        client_text: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Converts speech audio bytes into text with detected language.
        Prioritizes real STT (Groq/OpenAI Whisper), gracefully falls back to client-assisted
        transcription or deterministic mock audio tags. Never returns hardcoded demo text.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            if client_text and client_text.strip():
                c_text = client_text.strip()
                return TranscriptionResult(
                    text=c_text,
                    language=language or LanguageDetector.detect_language(c_text),
                    confidence=0.95,
                    provider="client"
                )
            return TranscriptionResult(
                text="",
                language="en",
                confidence=0.0,
                provider="none"
            )

        detected_filename, mime_type = self.detect_audio_container(audio_bytes)
        file_to_send = filename or detected_filename

        # 1. Groq Cloud Whisper API (ultra-fast transcription)
        if self.groq_api_key:
            try:
                headers = {"Authorization": f"Bearer {self.groq_api_key}"}
                files = {"file": (file_to_send, audio_bytes, mime_type)}
                data = {"model": settings.STT_MODEL}
                if language:
                    data["language"] = language

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data
                    )
                    if resp.status_code == 200:
                        res_json = resp.json()
                        text = res_json.get("text", "").strip()
                        if text:
                            detected_lang = language or LanguageDetector.detect_language(text)
                            return TranscriptionResult(
                                text=text,
                                language=detected_lang,
                                confidence=0.95,
                                provider="groq"
                            )
            except Exception:
                pass  # Fall through to fallback

        # 2. OpenAI Whisper API
        if self.openai_api_key:
            try:
                headers = {"Authorization": f"Bearer {self.openai_api_key}"}
                files = {"file": (file_to_send, audio_bytes, mime_type)}
                data = {"model": "whisper-1"}
                if language:
                    data["language"] = language

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data
                    )
                    if resp.status_code == 200:
                        res_json = resp.json()
                        text = res_json.get("text", "").strip()
                        if text:
                            detected_lang = language or LanguageDetector.detect_language(text)
                            return TranscriptionResult(
                                text=text,
                                language=detected_lang,
                                confidence=0.95,
                                provider="openai"
                            )
            except Exception:
                pass  # Fall through to fallback

        # 3. Client-assisted transcription (from browser Web Speech API / client)
        if client_text and client_text.strip():
            c_text = client_text.strip()
            detected_lang = language or LanguageDetector.detect_language(c_text)
            return TranscriptionResult(
                text=c_text,
                language=detected_lang,
                confidence=0.95,
                provider="browser_speech_recognition"
            )

        # 4. Deterministic Mock/Fallback Transcriber
        try:
            # Check if mock audio contains a text tag (e.g. for deterministic unit testing)
            text_candidate = audio_bytes.decode("utf-8", errors="ignore").strip()
            if text_candidate.startswith("TEXT:"):
                extracted_text = text_candidate[5:].strip()
                detected_lang = language or LanguageDetector.detect_language(extracted_text)
                return TranscriptionResult(
                    text=extracted_text,
                    language=detected_lang,
                    confidence=1.0,
                    provider="mock"
                )
        except Exception:
            pass

        # If audio cannot be decoded and no client transcript is provided, return empty
        return TranscriptionResult(
            text="",
            language=language or "en",
            confidence=0.0,
            provider="mock"
        )
