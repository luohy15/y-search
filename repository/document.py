from sqlalchemy.orm import Session
from sqlalchemy import text

from entity.document import Document


def search_keyword(
    session: Session,
    query: str,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    """
    Full-text keyword search using PostgreSQL tsvector.
    Returns list of (Document, rank_score) tuples.
    """
    search_query = text("""
        SELECT
            d.*,
            ts_rank_cd(d.content_tsvector, plainto_tsquery('english', :query)) as rank
        FROM documents d
        WHERE d.content_tsvector @@ plainto_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :limit
    """)

    results = session.execute(search_query, {"query": query, "limit": limit}).fetchall()

    docs = []
    for row in results:
        doc = session.query(Document).filter(Document.id == row.id).first()
        if doc:
            docs.append((doc, row.rank))
    return docs
