import pytest
import os
import tempfile
import gc
from app.storage.database import DatabaseManager
from app.storage.models import Lead, LeadStatus, SentimentType
from app.memory.buffer import ConversationMemory
from app.intelligence.sentiment import SentimentAnalyzer
from app.intelligence.lead_qualifier import LeadQualifier
from app.rag.retriever import RAGRetriever


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(path)
    db.init_db()
    yield db
    gc.collect()
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        pass


def test_contextual_memory_sliding_window(temp_db):
    """Verify that ConversationMemory enforces max_turns sliding window and maintains order."""
    mem = ConversationMemory(session_id="mem-session-1", max_turns=4, db=temp_db)
    for i in range(6):
        mem.add_turn(speaker="user" if i % 2 == 0 else "agent", text=f"Message {i}")

    assert len(mem.turns) == 4
    assert mem.turns[0].text == "Message 2"
    assert mem.turns[-1].text == "Message 5"

    history = mem.get_formatted_history()
    assert "Message 2" in history
    assert "Message 5" in history
    assert "Message 0" not in history


def test_contextual_memory_db_sync(temp_db):
    """Verify state hydration and synchronization with SQLite persistence."""
    mem = ConversationMemory(session_id="mem-session-2", db=temp_db)
    mem.entities.name = "David Miller"
    mem.entities.company = "Apex Systems"
    mem.entities.lead_score = 75
    mem.entities.qualification_status = LeadStatus.QUALIFIED
    mem.sync_to_db()

    # Hydrate new memory instance from same session
    hydrated_mem = ConversationMemory(session_id="mem-session-2", db=temp_db)
    assert hydrated_mem.entities.name == "David Miller"
    assert hydrated_mem.entities.company == "Apex Systems"
    assert hydrated_mem.entities.lead_score == 75
    assert hydrated_mem.entities.qualification_status == LeadStatus.QUALIFIED


def test_sentiment_analysis():
    """Verify sentiment categorization, scoring, and objection detection."""
    analyzer = SentimentAnalyzer()

    # Positive buying signal
    res_pos = analyzer.analyze("This sounds great, exactly what we need to book a demo!")
    assert res_pos.sentiment == SentimentType.POSITIVE
    assert res_pos.score > 0.3

    # Hesitant / budget concern
    res_hes = analyzer.analyze("I am not sure, your solution seems too expensive for our budget.")
    assert res_hes.sentiment == SentimentType.HESITANT
    assert res_hes.is_objection is True
    assert "budget_objection" in res_hes.detected_emotions

    # Frustrated
    res_fru = analyzer.analyze("I am extremely frustrated, let me speak to a human right now!")
    assert res_fru.sentiment == SentimentType.FRUSTRATED
    assert "frustrated" in res_fru.detected_emotions

    # Neutral
    res_neu = analyzer.analyze("Can you share what integrations you support?")
    assert res_neu.sentiment == SentimentType.NEUTRAL


def test_lead_qualification_bant_progression(temp_db):
    """
    Simulate a multi-turn conversation where prospect provides BANT attributes step-by-step
    and verify incremental lead score progression.
    """
    qualifier = LeadQualifier()
    analyzer = SentimentAnalyzer()
    lead = Lead(session_id="bant-session-1")

    # Turn 1: Introduction and Need
    t1 = "Hi, my name is Priya Sharma from CloudFlow Inc. We need an autonomous inbound sales agent because we miss leads after hours."
    s1 = analyzer.analyze(t1)
    lead = qualifier.evaluate_turn(lead, t1, s1)

    assert lead.name == "Priya Sharma"
    assert lead.company == "CloudFlow Inc"
    assert lead.need is not None
    assert lead.lead_score >= 25
    assert lead.qualification_status == LeadStatus.QUALIFYING

    # Turn 2: Authority
    t2 = "I am the VP of Sales and have the final authority to evaluate solutions."
    s2 = analyzer.analyze(t2)
    lead = qualifier.evaluate_turn(lead, t2, s2)

    assert lead.authority == "Vp Of Sales"
    assert lead.lead_score >= 50
    assert lead.qualification_status == LeadStatus.QUALIFIED

    # Turn 3: Budget & Timeline
    t3 = "We have an approved budget of $15,000 and want to deploy immediately this month."
    s3 = analyzer.analyze(t3)
    lead = qualifier.evaluate_turn(lead, t3, s3)

    assert lead.budget is not None
    assert lead.timeline is not None
    # All 4 BANT criteria satisfied + positive sentiment bonus
    assert lead.lead_score >= 80
    assert lead.qualification_status == LeadStatus.HOT_LEAD


def test_disqualification(temp_db):
    """Verify that an explicit rejection immediately marks the lead as DISQUALIFIED."""
    qualifier = LeadQualifier()
    analyzer = SentimentAnalyzer()
    lead = Lead(session_id="disqualify-session", lead_score=50, qualification_status=LeadStatus.QUALIFIED)

    t = "We are not interested, please remove me and stop calling."
    s = analyzer.analyze(t)
    lead = qualifier.evaluate_turn(lead, t, s)

    assert lead.qualification_status == LeadStatus.DISQUALIFIED
    assert lead.lead_score == 0


def test_conversation_memory_with_rag_integration(temp_db):
    """
    Verify complete integration loop:
    User asks a question -> Sentiment analyzed -> BANT evaluated -> Memory updated -> RAG retrieved.
    """
    memory = ConversationMemory(session_id="integ-session-1", db=temp_db)
    analyzer = SentimentAnalyzer()
    qualifier = LeadQualifier()
    retriever = RAGRetriever()

    user_query = "Hello, I am Robert from ScaleCorp. What is the pricing for your growth tier? We want to deploy in Q1."

    # 1. Sentiment
    sentiment = analyzer.analyze(user_query)
    assert sentiment.sentiment in [SentimentType.NEUTRAL, SentimentType.POSITIVE]

    # 2. Contextual Memory
    turn = memory.add_turn(speaker="user", text=user_query, sentiment=sentiment.sentiment)
    assert len(memory.turns) == 1

    # 3. Lead Qualification
    current_lead = memory.db.get_lead_by_session(memory.session_id) or Lead(session_id=memory.session_id)
    updated_lead = qualifier.evaluate_turn(current_lead, user_query, sentiment)

    # 4. Sync to DB and Memory
    memory.entities.name = updated_lead.name
    memory.entities.company = updated_lead.company
    memory.entities.timeline = updated_lead.timeline
    memory.entities.budget = updated_lead.budget
    memory.entities.lead_score = updated_lead.lead_score
    memory.entities.qualification_status = updated_lead.qualification_status
    memory.sync_to_db()

    # 5. RAG Retrieval for the query
    rag_results = retriever.retrieve(user_query, top_k=2)
    assert len(rag_results) > 0
    formatted_context = retriever.format_for_prompt(rag_results)

    assert "Growth Tier: $1,499/month" in formatted_context
    assert updated_lead.name == "Robert"
    assert updated_lead.timeline in ["Q1", "in Q1"]
    assert memory.db.get_lead_by_session("integ-session-1").name == "Robert"


def test_lead_scoring_threshold_boundaries():
    """Verify exact BANT scoring boundaries and status mapping."""
    qualifier = LeadQualifier()
    analyzer = SentimentAnalyzer()
    neutral_sent = analyzer.analyze("Hello")

    # 0 points -> UNQUALIFIED
    l0 = qualifier.evaluate_turn(Lead(session_id="s0"), "Hello", neutral_sent)
    assert l0.lead_score == 0
    assert l0.qualification_status == LeadStatus.UNQUALIFIED

    # 25 points -> QUALIFYING (1 criterion: need)
    l1 = qualifier.evaluate_turn(Lead(session_id="s1"), "We need an automated sales agent", neutral_sent)
    assert l1.lead_score == 25
    assert l1.qualification_status == LeadStatus.QUALIFYING

    # 50 points -> QUALIFIED (2 criteria: authority + need)
    l2 = Lead(session_id="s2", authority="VP of Sales")
    l2 = qualifier.evaluate_turn(l2, "We need an automated sales agent", neutral_sent)
    assert l2.lead_score == 50
    assert l2.qualification_status == LeadStatus.QUALIFIED

    # 75 points -> HOT_LEAD (3 criteria: authority + need + timeline)
    l3 = Lead(session_id="s3", authority="VP of Sales", need="Automate leads")
    l3 = qualifier.evaluate_turn(l3, "We want to launch this month", neutral_sent)
    assert l3.lead_score == 75
    assert l3.qualification_status == LeadStatus.HOT_LEAD

    # 100 points -> HOT_LEAD (4 criteria: budget + authority + need + timeline)
    l4 = Lead(session_id="s4", authority="VP of Sales", need="Automate leads", timeline="this month")
    l4 = qualifier.evaluate_turn(l4, "We have a budget of $10,000", neutral_sent)
    assert l4.lead_score == 100
    assert l4.qualification_status == LeadStatus.HOT_LEAD
