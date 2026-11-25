from typing import Optional, List, Dict, Any
from app.storage.database import DatabaseManager, get_db
from app.storage.models import CallLog, CallTurn, Lead, Appointment, SentimentType
from app.analytics.summarizer import CallSummarizer, SummaryResult


class BusinessInsightExtractor:
    """
    Business Insights Engine.
    Extracts high-value sales intelligence from completed calls:
    - BANT completeness & lead fit analysis
    - Sentiment trajectory and trend analysis
    - Objection cataloging
    - Sales conversion probability
    - Recommended commercial next steps
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or get_db()
        self.summarizer = CallSummarizer()

    async def generate_call_analytics(
        self,
        session_id: str,
        turns: List[CallTurn],
        lead: Optional[Lead] = None,
        appointment: Optional[Appointment] = None,
        duration_seconds: int = 0
    ) -> CallLog:
        """
        Processes a finished call session, extracts business insights,
        generates the executive summary, and persists the CallLog into SQLite.
        """
        if not lead:
            lead = self.db.get_lead_by_session(session_id)

        # 1. Sentiment trajectory & detected objections
        sentiment_trajectory: List[str] = []
        detected_objections: List[str] = []

        for turn in turns:
            if turn.sentiment:
                sentiment_trajectory.append(turn.sentiment.value)
            if turn.speaker == "user":
                t_lower = turn.text.lower()
                if any(w in t_lower for w in ["expensive", "budget", "costly"]):
                    detected_objections.append("budget_objection")
                if any(w in t_lower for w in ["human", "replace", "in-house", "another vendor"]):
                    detected_objections.append("competition_objection")
                if any(w in t_lower for w in ["safe", "security", "leak", "soc2", "privacy"]):
                    detected_objections.append("security_concern")
                if any(w in t_lower for w in ["not now", "later", "busy", "next quarter"]):
                    detected_objections.append("timing_objection")

        detected_objections = list(set(detected_objections))

        # 2. Sentiment Trend
        sentiment_trend = "STABLE"
        if len(sentiment_trajectory) >= 2:
            first_half = sentiment_trajectory[:len(sentiment_trajectory)//2]
            second_half = sentiment_trajectory[len(sentiment_trajectory)//2:]
            pos_first = first_half.count("POSITIVE")
            pos_second = second_half.count("POSITIVE")
            if pos_second > pos_first:
                sentiment_trend = "IMPROVING"
            elif second_half.count("NEGATIVE") > first_half.count("NEGATIVE"):
                sentiment_trend = "DECLINING"

        # 3. Conversion Probability Calculation
        conversion_prob = 10
        score = lead.lead_score if lead else 0

        if lead:
            if lead.qualification_status.value == "HOT_LEAD":
                conversion_prob = 85 if appointment else 70
            elif lead.qualification_status.value == "QUALIFIED":
                conversion_prob = 55
            elif lead.qualification_status.value == "QUALIFYING":
                conversion_prob = 30
            elif lead.qualification_status.value == "DISQUALIFIED":
                conversion_prob = 0

            # Bonus if positive sentiment was dominant
            if sentiment_trajectory.count("POSITIVE") > len(sentiment_trajectory) / 2 and conversion_prob > 0:
                conversion_prob = min(95, conversion_prob + 10)

        # 4. Generate Executive Summary
        summary_res = await self.summarizer.summarize_call(turns, lead)

        # 5. Assemble Structured Sales Insights
        sales_insights: Dict[str, Any] = {
            "lead_fit": "EXCELLENT" if score >= 75 else ("GOOD" if score >= 50 else "MODERATE"),
            "conversion_probability_percent": conversion_prob,
            "sentiment_trend": sentiment_trend,
            "primary_objections": detected_objections,
            "appointment_scheduled": bool(appointment),
            "appointment_time": appointment.start_time if appointment else None,
            "bant_breakdown": {
                "budget_confirmed": bool(lead and lead.budget),
                "authority_confirmed": bool(lead and lead.authority),
                "need_confirmed": bool(lead and lead.need),
                "timeline_confirmed": bool(lead and lead.timeline),
                "total_score": score
            },
            "commercial_recommendation": (
                "High-intent opportunity: Proceed with live demonstration and proposal."
                if conversion_prob >= 70 else (
                    "Nurture opportunity: Provide collateral and schedule follow-up."
                    if conversion_prob >= 30 else "Low priority / Disqualified."
                )
            )
        }

        # 6. Save CallLog Record
        call_log = CallLog(
            session_id=session_id,
            lead_id=lead.id if lead else None,
            transcript=turns,
            summary=summary_res.summary,
            key_takeaways=summary_res.key_takeaways,
            action_items=summary_res.action_items,
            sentiment_trajectory=sentiment_trajectory,
            detected_objections=detected_objections,
            sales_insights=sales_insights,
            duration_seconds=duration_seconds
        )

        saved_log = self.db.save_call_log(call_log)
        return saved_log
