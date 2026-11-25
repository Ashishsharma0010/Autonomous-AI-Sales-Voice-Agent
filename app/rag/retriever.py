import json
import os
from typing import List, Optional
from pathlib import Path
from app.rag.vector_store import VectorStore, Document, SearchResult


class RAGRetriever:
    """
    RAG Knowledge Retriever for B2B AI Sales Agent.
    Manages loading of product catalogs, enterprise FAQs, and objection handling guidelines.
    Supplies context chunks for LLM sales grounding and objection resolution.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent / "data"

        self.vector_store = VectorStore()
        self._is_loaded = False
        self.load_knowledge_base()

    def load_knowledge_base(self) -> int:
        """Loads and indexes all documents from the knowledge repository."""
        self.vector_store.clear()
        docs: List[Document] = []

        # 1. Load Products
        products_file = self.data_dir / "products.json"
        if products_file.exists():
            with open(products_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    docs.append(Document(
                        id=item["id"],
                        title=item.get("title", ""),
                        category=item.get("category", "product"),
                        content=item["content"],
                        metadata=item.get("metadata", {})
                    ))

        # 2. Load FAQs
        faqs_file = self.data_dir / "faqs.json"
        if faqs_file.exists():
            with open(faqs_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    docs.append(Document(
                        id=item["id"],
                        title=item.get("question", "FAQ"),
                        category="faq",
                        content=f"Question: {item.get('question', '')}\nAnswer: {item['content']}",
                        metadata=item.get("metadata", {})
                    ))

        # 3. Load Objections
        objections_file = self.data_dir / "objections.json"
        if objections_file.exists():
            with open(objections_file, "r", encoding="utf-8") as f:
                items = json.load(f)
                for item in items:
                    docs.append(Document(
                        id=item["id"],
                        title=f"Objection: {item.get('objection', '')}",
                        category="objection",
                        content=f"Objection: {item.get('objection', '')}\n{item['content']}",
                        metadata=item.get("metadata", {})
                    ))

        self.vector_store.add_documents(docs)
        self._is_loaded = True
        return len(docs)

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.12
    ) -> List[SearchResult]:
        """
        Retrieves relevant knowledge snippets for a sales query.
        Optionally filters by document category ('product', 'pricing', 'faq', 'objection').
        """
        if not self._is_loaded:
            self.load_knowledge_base()

        results = self.vector_store.search(query=query, top_k=top_k * 2, threshold=threshold)

        if category:
            results = [r for r in results if r.document.category == category]

        return results[:top_k]

    def format_for_prompt(self, results: List[SearchResult]) -> str:
        """Formats retrieved chunks into clean markdown context for the LLM prompt."""
        if not results:
            return ""

        chunks = []
        for i, res in enumerate(results, start=1):
            doc = res.document
            title = doc.title or doc.id
            chunks.append(f"[{i}] {title} (Category: {doc.category}, Relevance: {int(res.score * 100)}%):\n{doc.content}")

        return "\n\n".join(chunks)


# Singleton instance
_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
