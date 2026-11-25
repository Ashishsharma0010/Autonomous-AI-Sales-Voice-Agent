import re
from typing import Optional, Dict, Any, Tuple
from app.storage.models import Lead, LeadStatus, SentimentType
from app.intelligence.sentiment import SentimentResult


class LeadQualifier:
    """
    Intelligent Lead Qualification Engine using the BANT Framework:
    - Budget (25 points): Dedicated spend, pricing range, or approved capital.
    - Authority (25 points): C-suite, VP, Director, or authorized decision maker.
    - Need (25 points): Concrete business pain point, inefficiency, or scaling goal.
    - Timeline (25 points): Implementation roadmap (ASAP, next month, Q1/Q2, this quarter).
    
    Evaluates customer utterances, updates lead scoring (0-100), and assigns qualification status.
    """

    # BANT extraction patterns
    BUDGET_PATTERNS = [
        r"(?:\$\s*|\busd\s*)\d+[\d,]*(?:\.\d+)?|\b(?:\d+k|\d+\s?thousand|\d+\s?hundred|\d+\s?dollars)\b",
        r"\b(allocated budget|spend around|afford|pricing tier|starter tier|growth tier|enterprise tier|budget)\b"
    ]

    AUTHORITY_PATTERNS = [
        r"\b(chief [a-z]+ officer|vp of [a-z]+|vice president of [a-z]+|head of [a-z]+|director of [a-z]+|[a-z]+ manager|manager|ceo|cto|cfo|cro|cmo|coo|vp|vice president|director|founder|co-founder|decision maker|i approve|my budget|owner|managing director)\b",
        r"\b(i make the decision|final say|evaluating for my team)\b"
    ]

    NEED_PATTERNS = [
        r"\b(need|looking for|require|automate|pain point|qualify leads|missed calls|missed leads|losing leads|sales efficiency|after\s+(?:business\s+)?hours|inbound sdr|outbound pipeline|sdr capacity)\b",
        r"\b(too many leads|cannot keep up|scaling our team|reduce cost)\b"
    ]

    TIMELINE_PATTERNS = [
        r"\b(immediately|asap|this month|next month|in \d+ weeks?|q[1-4]|this quarter|next quarter|within 30 days|urgent|evaluating right now)\b"
    ]

    # Contact extraction patterns
    EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"
    NAME_PATTERNS = [
        r"(?:my name is|i am|this is|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)(?:[\,\.]|\s+(?:from|at|with|and)|$)",
        r"^([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:here|speaking)"
    ]
    COMPANY_PATTERNS = [
        r"(?:at|with|from|represent|company is)\s+([A-Z0-9][A-Za-z0-9\s&]+?)(?:\.|\,|$|\s+and|\s+we)",
    ]

    def extract_entities(self, text: str) -> Dict[str, Optional[str]]:
        """Extracts contact credentials and BANT attributes from user speech."""
        extracted: Dict[str, Optional[str]] = {}

        # 1. Email
        email_match = re.search(self.EMAIL_PATTERN, text, re.IGNORECASE)
        if email_match:
            extracted["email"] = email_match.group(0)

        # 2. Name
        for np in self.NAME_PATTERNS:
            name_match = re.search(np, text, re.IGNORECASE)
            if name_match:
                raw_name = name_match.group(1).strip()
                cleaned_name = re.sub(r"\s+(from|at|with|and)$", "", raw_name, flags=re.IGNORECASE).strip()
                if cleaned_name.lower() not in ["from", "at", "with", "the", "a", "an"]:
                    extracted["name"] = cleaned_name.title()
                break

        # 3. Company
        for cp in self.COMPANY_PATTERNS:
            comp_match = re.search(cp, text)
            if comp_match:
                clean_comp = comp_match.group(1).strip()
                if len(clean_comp) > 2 and clean_comp.lower() not in ["the", "a", "an"]:
                    extracted["company"] = clean_comp
                break

        # 4. Budget
        for bp in self.BUDGET_PATTERNS:
            b_match = re.search(bp, text, re.IGNORECASE)
            if b_match:
                extracted["budget"] = b_match.group(0).strip()
                break

        # 5. Authority
        for ap in self.AUTHORITY_PATTERNS:
            a_match = re.search(ap, text, re.IGNORECASE)
            if a_match:
                extracted["authority"] = a_match.group(0).strip().title()
                break

        # 6. Need
        for nep in self.NEED_PATTERNS:
            n_match = re.search(nep, text, re.IGNORECASE)
            if n_match:
                extracted["need"] = text.strip()
                break

        # 7. Timeline
        for tp in self.TIMELINE_PATTERNS:
            t_match = re.search(tp, text, re.IGNORECASE)
            if t_match:
                extracted["timeline"] = t_match.group(0).strip()
                break

        return extracted

    def evaluate_turn(
        self,
        lead: Lead,
        user_text: str,
        sentiment: SentimentResult
    ) -> Lead:
        """
        Updates lead BANT attributes from conversational turn,
        recalculates lead score, and assigns updated qualification status.
        """
        extracted = self.extract_entities(user_text)

        # Update contact credentials if found
        if extracted.get("name") and not lead.name:
            lead.name = extracted["name"]
        if extracted.get("email") and not lead.email:
            lead.email = extracted["email"]
        if extracted.get("company") and not lead.company:
            lead.company = extracted["company"]

        # Update BANT fields
        if extracted.get("budget"):
            lead.budget = extracted["budget"]
        if extracted.get("authority"):
            lead.authority = extracted["authority"]
            if not lead.role:
                lead.role = extracted["authority"]
        if extracted.get("need") and not lead.need:
            lead.need = extracted["need"]
        if extracted.get("timeline"):
            lead.timeline = extracted["timeline"]

        # Check for explicit disqualification
        disqualify_cues = ["not interested", "stop calling", "remove me", "don't call again", "unsubscribe"]
        if any(cue in user_text.lower() for cue in disqualify_cues):
            lead.qualification_status = LeadStatus.DISQUALIFIED
            lead.lead_score = 0
            return lead

        # Recalculate BANT score
        # Scoring Matrix:
        # - Budget: 25 points
        # - Authority: 25 points
        # - Need: 25 points
        # - Timeline: 25 points
        # - Sentiment modifier: +10 for POSITIVE, -15 for FRUSTRATED
        score = 0
        if lead.budget:
            score += 25
        if lead.authority:
            score += 25
        if lead.need:
            score += 25
        if lead.timeline:
            score += 25

        # Sentiment bonus / deduction
        if sentiment.sentiment == SentimentType.POSITIVE and score > 0:
            score = min(100, score + 10)
        elif sentiment.sentiment == SentimentType.FRUSTRATED:
            score = max(0, score - 15)

        lead.lead_score = score

        # Qualification Status Mapping:
        # - 75 to 100: HOT_LEAD (Ready for immediate demonstration booking)
        # - 50 to 74:  QUALIFIED (2 criteria met; consultative qualification continuing)
        # - 25 to 49:  QUALIFYING (1 criterion met)
        # - 0 to 24:   UNQUALIFIED (0 criteria met)
        if score >= 75:
            lead.qualification_status = LeadStatus.HOT_LEAD
        elif score >= 50:
            lead.qualification_status = LeadStatus.QUALIFIED
        elif score >= 25:
            lead.qualification_status = LeadStatus.QUALIFYING
        else:
            lead.qualification_status = LeadStatus.UNQUALIFIED

        return lead
