import pytest
from app.rag.vector_store import VectorStore, Document
from app.rag.retriever import RAGRetriever, get_retriever


@pytest.fixture
def retriever():
    """Provides an initialized RAGRetriever instance loaded with sales catalog."""
    return RAGRetriever()


def test_knowledge_base_loading(retriever):
    """Verify all knowledge categories (products, FAQs, objections) load correctly."""
    assert len(retriever.vector_store.documents) >= 10
    categories = {doc.category for doc in retriever.vector_store.documents}
    assert "product" in categories
    assert "pricing" in categories
    assert "faq" in categories
    assert "objection" in categories


def test_pricing_retrieval(retriever):
    """Verify accurate retrieval for pricing and tier inquiries."""
    results = retriever.retrieve("How much does Fortivex cost per month for the starter tier?", top_k=2)
    assert len(results) > 0
    top_doc = results[0].document
    assert top_doc.category == "pricing" or "pricing" in (top_doc.title or "").lower()
    assert "$499" in top_doc.content
    assert results[0].score > 0.15


def test_security_compliance_retrieval(retriever):
    """Verify security, SOC2, and GDPR inquiries retrieve the security FAQ."""
    results = retriever.retrieve("Is customer data SOC2 compliant and encrypted?", top_k=2)
    assert len(results) > 0
    contents = " ".join([r.document.content for r in results])
    assert "SOC2 Type II" in contents
    assert "AES-256" in contents


def test_objection_handling_retrieval(retriever):
    """Verify retrieval of objection handling response when prospect objects to cost."""
    results = retriever.retrieve("Your product is too expensive compared to human SDRs", top_k=2)
    assert len(results) > 0
    objection_results = [r for r in results if r.document.category == "objection"]
    assert len(objection_results) > 0
    assert "80% reduction" in objection_results[0].document.content


def test_multilingual_faq_retrieval(retriever):
    """Verify multilingual capability inquiry retrieves the multilingual FAQ."""
    results = retriever.retrieve("Does the sales agent speak Spanish, Hindi, or French?", top_k=2)
    assert len(results) > 0
    contents = " ".join([r.document.content for r in results])
    assert "40 languages" in contents or "multilingual" in contents.lower()


def test_empty_or_irrelevant_query_handling(retriever):
    """Verify that irrelevant queries return no results or empty list when below threshold."""
    # Blank query
    empty_res = retriever.retrieve("", top_k=3)
    assert empty_res == []

    # Gibberish / irrelevant query
    irrelevant_res = retriever.retrieve("xyzqwerty astronaut mars rocket orbit 998877", top_k=3, threshold=0.3)
    assert irrelevant_res == []


def test_prompt_formatting(retriever):
    """Verify formatted string generation for LLM context injection."""
    results = retriever.retrieve("latency and response time", top_k=1)
    assert len(results) == 1
    prompt_chunk = retriever.format_for_prompt(results)
    assert "[1]" in prompt_chunk
    assert "Relevance:" in prompt_chunk
    assert "<400" in prompt_chunk or "latency" in prompt_chunk.lower()
