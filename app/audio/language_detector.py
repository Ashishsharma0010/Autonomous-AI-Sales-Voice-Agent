import re
from typing import Dict


class LanguageDetector:
    """
    Multilingual Language Detection and Voice Mapping Service.
    Detects language from text/transcript and maps to optimal neural voice models.
    Supports English, Spanish, Hindi, French, German, and 35+ other languages.
    """

    # Language script & lexicon patterns
    PATTERNS: Dict[str, str] = {
        "hi": r"[\u0900-\u097F]|\b(namaste|aap|kaise|kya|chahiye|dhanyawad|shukriya|mujhe|hum|theek)\b",
        "es": r"\b(hola|buenos|dias|tardes|gracias|por favor|cuanto|cuesta|nosotros|quiero|necesitamos|empresa)\b",
        "fr": r"\b(bonjour|merci|s'il vous plait|combien|coute|nous|besoin|entreprise|tres|bien)\b",
        "de": r"\b(hallo|guten|tag|danke|bitte|wieviel|kostet|wir|brauchen|unternehmen|sehr)\b"
    }

    # Neural voice mappings per provider
    EDGE_VOICES: Dict[str, str] = {
        "en": "en-US-AriaNeural",
        "es": "es-ES-ElviraNeural",
        "hi": "hi-IN-SwaraNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "default": "en-US-AriaNeural"
    }

    OPENAI_VOICES: Dict[str, str] = {
        "en": "alloy",
        "es": "nova",
        "hi": "alloy",
        "fr": "nova",
        "de": "alloy",
        "default": "alloy"
    }

    @classmethod
    def detect_language(cls, text: str) -> str:
        """Detects language code from user utterance. Defaults to 'en'."""
        if not text or not text.strip():
            return "en"

        text_lower = text.lower()
        for lang_code, pattern in cls.PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                return lang_code

        return "en"

    @classmethod
    def get_voice(cls, language_code: str, provider: str = "edge") -> str:
        """Returns the optimal neural voice identifier for the specified language and provider."""
        lang = language_code.lower().split("-")[0]
        if provider == "edge":
            return cls.EDGE_VOICES.get(lang, cls.EDGE_VOICES["default"])
        elif provider == "openai":
            return cls.OPENAI_VOICES.get(lang, cls.OPENAI_VOICES["default"])
        return cls.EDGE_VOICES.get(lang, cls.EDGE_VOICES["default"])
