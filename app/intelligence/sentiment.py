import re
from typing import Tuple, List
from pydantic import BaseModel
from app.storage.models import SentimentType


class SentimentResult(BaseModel):
    sentiment: SentimentType
    score: float  # -1.0 (strongly negative) to +1.0 (strongly positive)
    is_objection: bool
    detected_emotions: List[str]


class SentimentAnalyzer:
    """
    Real-Time Conversation Sentiment & Objection Analyzer.
    Provides sub-millisecond lexical and semantic tone scoring tailored to sales conversations.
    Detects positive buying signals, price and timing hesitation, frustration, and objections.
    """

    POSITIVE_SIGNALS = [
        "great", "excellent", "awesome", "perfect", "sounds good", "impressive",
        "love it", "definitely", "interested", "exactly what we need", "excited",
        "fantastic", "schedule a demo", "book a call", "sign up", "let's do it",
        "yes", "absolutely", "helpful", "reasonable", "makes sense"
    ]

    HESITANT_SIGNALS = [
        "not sure", "maybe", "depends", "might be", "let me think", "expensive",
        "costly", "need to check", "consult with team", "hesitant", "skeptical",
        "worried", "concerned", "seems high", "not ready"
    ]

    NEGATIVE_SIGNALS = [
        "dislike", "bad", "poor", "unacceptable", "don't want", "not interested",
        "waste of time", "pass", "no thanks", "terrible", "useless", "won't work"
    ]

    FRUSTRATED_SIGNALS = [
        "angry", "annoyed", "frustrated", "stop calling", "ridiculous", "hate this",
        "robot", "speak to a human", "terrible experience", "let me talk to someone"
    ]

    OBJECTION_PATTERNS = [
        (r"\b(too expensive|pricing is high|expensive pricing|costly|budget is tight|out of budget|can't afford)\b", "budget_objection"),
        (r"\b(already have|current vendor|existing team|use another|in-house)\b", "competition_objection"),
        (r"\b(not now|later|next quarter|next year|no time|too busy)\b", "timing_objection"),
        (r"\b(robotic|sound like a bot|ai voice|fake voice|not a real person)\b", "quality_objection"),
        (r"\b(security leak|data leak|is it safe|privacy leak)\b", "security_concern")
    ]

    @staticmethod
    def _contains_word_or_phrase(pattern: str, text: str) -> bool:
        return bool(re.search(rf"\b{re.escape(pattern)}\b", text, re.IGNORECASE))

    def analyze(self, text: str) -> SentimentResult:
        """Analyzes text utterance and returns structured sentiment result."""
        text_lower = text.lower().strip()
        if not text_lower:
            return SentimentResult(
                sentiment=SentimentType.NEUTRAL,
                score=0.0,
                is_objection=False,
                detected_emotions=["neutral"]
            )

        pos_count = sum(1 for phrase in self.POSITIVE_SIGNALS if self._contains_word_or_phrase(phrase, text_lower))
        hes_count = sum(1 for phrase in self.HESITANT_SIGNALS if self._contains_word_or_phrase(phrase, text_lower))
        neg_count = sum(1 for phrase in self.NEGATIVE_SIGNALS if self._contains_word_or_phrase(phrase, text_lower))
        fru_count = sum(1 for phrase in self.FRUSTRATED_SIGNALS if self._contains_word_or_phrase(phrase, text_lower))

        # Check explicit isolated "no"
        if re.search(r"\bno\b", text_lower) and not re.search(r"\b(no problem|no worries|no doubt|no thanks)\b", text_lower):
            neg_count += 1

        # Check objections
        is_objection = False
        detected_emotions: List[str] = []

        for pattern, obj_type in self.OBJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                is_objection = True
                detected_emotions.append(obj_type)

        # Calculate weighted score from -1.0 to 1.0
        score = (pos_count * 0.4) - (hes_count * 0.25) - (neg_count * 0.5) - (fru_count * 0.75)
        score = max(-1.0, min(1.0, score))

        if fru_count > 0:
            sentiment = SentimentType.FRUSTRATED
            detected_emotions.append("frustrated")
        elif neg_count > 0 and neg_count >= pos_count:
            sentiment = SentimentType.NEGATIVE
            detected_emotions.append("negative")
        elif hes_count > 0 or is_objection:
            sentiment = SentimentType.HESITANT
            detected_emotions.append("hesitant")
        elif pos_count > 0 and pos_count >= neg_count:
            sentiment = SentimentType.POSITIVE
            detected_emotions.append("positive")
        else:
            sentiment = SentimentType.NEUTRAL
            detected_emotions.append("neutral")

        return SentimentResult(
            sentiment=sentiment,
            score=round(score, 2),
            is_objection=is_objection,
            detected_emotions=list(set(detected_emotions))
        )
