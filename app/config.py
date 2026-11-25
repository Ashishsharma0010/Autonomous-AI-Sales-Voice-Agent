import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings with environment variable support.
    Configurable providers allow seamless toggling between free tiers (Groq, Edge-TTS)
    and commercial providers (OpenAI, Gemini, ElevenLabs).
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # LLM Settings
    LLM_PROVIDER: Literal["groq", "openai", "gemini"] = "groq"
    LLM_MODEL: str = "qwen/qwen3.8-27b"
    OPENAI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    GEMINI_API_KEY: str = Field(default="")

    # STT Settings
    STT_PROVIDER: Literal["groq", "openai", "browser"] = "groq"
    STT_MODEL: str = "whisper-large-v3-turbo"

    # TTS Settings
    TTS_PROVIDER: Literal["edge", "openai", "elevenlabs"] = "edge"
    TTS_VOICE: str = "en-US-AriaNeural"
    OPENAI_TTS_VOICE: str = "alloy"
    ELEVENLABS_API_KEY: str = Field(default="")
    ELEVENLABS_VOICE_ID: str = Field(default="")

    # Database
    DATABASE_PATH: str = "sales_agent.db"

    # Integrations
    CALENDAR_MODE: Literal["mock", "google"] = "mock"
    CRM_MODE: Literal["mock", "hubspot", "webhook"] = "mock"
    CRM_WEBHOOK_URL: str = Field(default="")

    @property
    def is_groq_available(self) -> bool:
        return bool(self.GROQ_API_KEY.strip())

    @property
    def is_openai_available(self) -> bool:
        return bool(self.OPENAI_API_KEY.strip())

    @property
    def is_gemini_available(self) -> bool:
        return bool(self.GEMINI_API_KEY.strip())


# Global settings singleton
settings = Settings()
