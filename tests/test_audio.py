import pytest
from app.audio.language_detector import LanguageDetector
from app.audio.stt import SpeechToTextService, TranscriptionResult
from app.audio.tts import TextToSpeechService, SpeechResult
from app.memory.buffer import ConversationMemory
from app.intelligence.sentiment import SentimentAnalyzer
from app.intelligence.lead_qualifier import LeadQualifier
from app.rag.retriever import RAGRetriever
from app.storage.database import DatabaseManager


def test_multilingual_language_detection():
    """Verify automatic language detection across English, Spanish, Hindi, French, and German."""
    # English
    assert LanguageDetector.detect_language("We want to schedule a live product demo for our team.") == "en"

    # Spanish
    assert LanguageDetector.detect_language("Hola, buenos dias, necesitamos una solucion para ventas de nuestra empresa.") == "es"

    # Hindi (Devanagari script & transliterated keywords)
    assert LanguageDetector.detect_language("नमस्ते, हमें अपनी कंपनी के लिए सेल्स ऑटोमेशन चाहिए") == "hi"
    assert LanguageDetector.detect_language("namaste, mujhe pricing janni hai") == "hi"

    # French
    assert LanguageDetector.detect_language("Bonjour, nous avons besoin d'un agent de vente pour notre entreprise.") == "fr"

    # German
    assert LanguageDetector.detect_language("Guten Tag, wieviel kostet das Starter Paket für unser Unternehmen?") == "de"


def test_multilingual_voice_selection():
    """Verify that language codes map to authentic regional neural voices."""
    assert LanguageDetector.get_voice("es", provider="edge") == "es-ES-ElviraNeural"
    assert LanguageDetector.get_voice("hi", provider="edge") == "hi-IN-SwaraNeural"
    assert LanguageDetector.get_voice("en", provider="edge") == "en-US-AriaNeural"
    assert LanguageDetector.get_voice("fr", provider="edge") == "fr-FR-DeniseNeural"
    assert LanguageDetector.get_voice("de", provider="edge") == "de-DE-KatjaNeural"


@pytest.mark.asyncio
async def test_stt_transcription():
    """Verify STT service handling of mock audio, text tags, and language detection."""
    stt = SpeechToTextService()

    # Empty audio check
    empty_res = await stt.transcribe(b"")
    assert empty_res.text == ""
    assert empty_res.confidence == 0.0

    # Mock audio tag with Spanish text
    spanish_audio = b"TEXT: Hola, queremos conocer el precio de Fortivex AI"
    res_es = await stt.transcribe(spanish_audio)
    assert "queremos conocer el precio" in res_es.text
    assert res_es.language == "es"
    assert res_es.confidence == 1.0


@pytest.mark.asyncio
async def test_tts_edge_neural_synthesis():
    """Verify Microsoft Edge Neural TTS synthesizes real audio bytes without requiring an API key."""
    tts = TextToSpeechService()
    tts.provider = "edge"

    # Synthesize English response
    res = await tts.synthesize("Welcome to Fortivex AI. How can I assist you today?", language="en")
    assert res is not None
    assert len(res.audio_bytes) > 2000  # Real audio bytes generated
    assert res.content_type == "audio/mpeg"
    assert res.voice_used == "en-US-AriaNeural"
    assert res.provider == "edge"


@pytest.mark.asyncio
async def test_tts_mock_fallback():
    """Verify that mock fallback produces valid WAV audio with standard 44-byte RIFF header."""
    tts = TextToSpeechService()
    tts.provider = "mock"

    res = await tts.synthesize("Fallback test audio message.")
    assert res.provider == "mock"
    assert res.content_type == "audio/wav"
    assert len(res.audio_bytes) >= 44
    assert res.audio_bytes[:4] == b"RIFF"
    assert res.audio_bytes[8:12] == b"WAVE"


def test_audio_container_detection():
    """Verify magic bytes detection for WebM, WAV, OGG, and MP3."""
    stt = SpeechToTextService()
    # WebM
    fn, mime = stt.detect_audio_container(b"\x1aE\xdf\xa3\x00\x00")
    assert mime == "audio/webm"
    assert fn == "recording.webm"

    # WAV
    fn, mime = stt.detect_audio_container(b"RIFF\x24\x00\x00\x00WAVE")
    assert mime == "audio/wav"
    assert fn == "recording.wav"

    # OGG
    fn, mime = stt.detect_audio_container(b"OggS\x00\x02")
    assert mime == "audio/ogg"


@pytest.mark.asyncio
async def test_audio_pipeline_full_integration():
    """
    Verify complete voice integration:
    Audio Ingestion -> STT -> Language & Sentiment -> Lead BANT Qualification -> RAG Grounding -> TTS Voice Matching.
    """
    stt = SpeechToTextService()
    tts = TextToSpeechService()
    analyzer = SentimentAnalyzer()
    qualifier = LeadQualifier()
    retriever = RAGRetriever()

    # 1. Incoming user audio
    mock_audio = b"TEXT: Hi, I am Carlos Mendez from InnovaTech. We need an autonomous inbound sales agent this month."
    stt_result = await stt.transcribe(mock_audio)
    assert stt_result.text != ""

    # 2. Sentiment & Language
    sentiment = analyzer.analyze(stt_result.text)
    assert stt_result.language == "en"

    # 3. Lead Qualification
    from app.storage.models import Lead, LeadStatus
    lead_obj = Lead(session_id="voice-integ-1")
    lead_obj = qualifier.evaluate_turn(lead_obj, stt_result.text, sentiment)
    assert lead_obj.name == "Carlos Mendez"
    assert lead_obj.company == "InnovaTech"
    assert lead_obj.qualification_status in [LeadStatus.QUALIFYING, LeadStatus.QUALIFIED]

    # 4. RAG Retrieval for the user's inquiry
    rag_docs = retriever.retrieve(stt_result.text, top_k=1)
    assert len(rag_docs) > 0

    # 5. TTS Response Generation with matched voice
    response_text = f"Hello Carlos, Fortivex Inbound SDR is built exactly for InnovaTech's sales automation."
    tts_result = await tts.synthesize(response_text, language=stt_result.language)
    assert len(tts_result.audio_bytes) > 0
    assert tts_result.voice_used == "en-US-AriaNeural"


@pytest.mark.asyncio
async def test_stt_client_assisted_transcription_and_no_hardcoded_fallback():
    """
    Verify that:
    1. STT uses client_text when external API is offline/mock without returning fake sentences.
    2. Real binary audio without API keys never defaults to 'Hello, I am interested in Fortivex AI...'.
    """
    stt = SpeechToTextService()
    stt.groq_api_key = ""
    stt.openai_api_key = ""

    # Real dummy webm binary header (not starting with TEXT:)
    raw_binary_audio = b"\x1aE\xdf\xa3\x01\x02\x03\x04"
    client_sentence = "Hi, my name is Rahul. I am the Sales Manager at TechNova. Can you explain what your AI voice agent does?"

    # Case 1: Raw audio with client_text supplied
    res_with_client = await stt.transcribe(raw_binary_audio, client_text=client_sentence)
    assert res_with_client.text == client_sentence
    assert "interested in Fortivex AI" not in res_with_client.text

    # Case 2: Raw audio without client_text and without TEXT: tag -> must NOT return hardcoded demo string
    res_raw_only = await stt.transcribe(raw_binary_audio)
    assert "interested in Fortivex AI" not in res_raw_only.text
    assert res_raw_only.text == ""
