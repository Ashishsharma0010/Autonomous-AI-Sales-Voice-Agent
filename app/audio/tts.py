import struct
from typing import Optional
from pydantic import BaseModel
import edge_tts
import httpx
from app.config import settings
from app.audio.language_detector import LanguageDetector


class SpeechResult(BaseModel):
    audio_bytes: bytes
    content_type: str
    voice_used: str
    provider: str


class TextToSpeechService:
    """
    Text-to-Speech (TTS) Service.
    Supports Microsoft Edge Neural TTS (zero-cost, 100% natural voices), OpenAI TTS, and Mock fallback.
    Automatically matches neural voice to the customer's conversational language.
    """

    def __init__(self):
        self.provider = settings.TTS_PROVIDER
        self.openai_api_key = settings.OPENAI_API_KEY

    @staticmethod
    def _create_mock_wav() -> bytes:
        """Generates valid 44-byte standard RIFF WAV silence for offline unit testing."""
        sample_rate = 16000
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        num_samples = 1600  # 100ms of audio
        data_size = num_samples * num_channels * bits_per_sample // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size
        )
        samples = b"\x00" * data_size
        return header + samples

    async def synthesize(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None
    ) -> SpeechResult:
        """
        Synthesizes text into high-fidelity neural audio.
        Matches voice to detected language.
        """
        if not text or not text.strip():
            return SpeechResult(
                audio_bytes=b"",
                content_type="audio/mpeg",
                voice_used="none",
                provider="none"
            )

        # 1. Edge Neural TTS (Free, high-fidelity neural voice, multilingual)
        if self.provider == "edge":
            target_voice = voice or LanguageDetector.get_voice(language, provider="edge")
            try:
                communicator = edge_tts.Communicate(text=text, voice=target_voice)
                chunks = []
                async for chunk in communicator.stream():
                    if chunk["type"] == "audio":
                        chunks.append(chunk["data"])
                audio_bytes = b"".join(chunks)

                if audio_bytes:
                    return SpeechResult(
                        audio_bytes=audio_bytes,
                        content_type="audio/mpeg",
                        voice_used=target_voice,
                        provider="edge"
                    )
            except Exception:
                pass  # Fallback to mock WAV

        # 2. OpenAI TTS
        if self.provider == "openai" and self.openai_api_key:
            target_voice = voice or LanguageDetector.get_voice(language, provider="openai")
            try:
                headers = {
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "tts-1",
                    "input": text,
                    "voice": target_voice
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/audio/speech",
                        headers=headers,
                        json=payload
                    )
                    if resp.status_code == 200:
                        return SpeechResult(
                            audio_bytes=resp.content,
                            content_type="audio/mpeg",
                            voice_used=target_voice,
                            provider="openai"
                        )
            except Exception:
                pass  # Fallback to mock WAV

        # 3. Mock Fallback Audio
        target_voice = voice or "mock-voice"
        return SpeechResult(
            audio_bytes=self._create_mock_wav(),
            content_type="audio/wav",
            voice_used=target_voice,
            provider="mock"
        )
