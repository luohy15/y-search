"""
Document search service.
"""
from loguru import logger

from config.database import get_db
from repository import document as document_repo


def _extract_snippet(content: str, query: str, max_length: int = 200) -> str:
    """Extract a relevant snippet from content around the query terms."""
    query_words = query.lower().split()
    content_lower = content.lower()

    best_pos = 0
    for word in query_words:
        pos = content_lower.find(word)
        if pos != -1:
            best_pos = pos
            break

    start = max(0, best_pos - 50)
    end = min(len(content), best_pos + max_length)

    snippet = content[start:end].strip()

    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."

    return snippet


def _format_search_result(doc, score: float, query: str) -> dict:
    """Format a document into a search result."""
    return {
        "id": doc.id,
        "filename": doc.filename,
        "title": doc.title,
        "snippet": _extract_snippet(doc.content_text, query),
        "score": round(score, 4),
    }


def search(query: str, limit: int = 10) -> dict:
    """
    Search documents using keyword-based full-text search.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        dict with query, items, and total
    """
    logger.info(f"Search: '{query}' (limit={limit})")

    with get_db() as session:
        results = document_repo.search_keyword(session, query, limit=limit)

        items = [
            _format_search_result(doc, score, query)
            for doc, score in results
        ]

        return {
            "query": query,
            "items": items,
            "total": len(items),
        }
