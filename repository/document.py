import re

from sqlalchemy.orm import Session
from sqlalchemy import text

from entity.document import Document

# CJK Unicode ranges (Chinese, Japanese, Korean)
_CJK_PATTERN = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')

# Pattern to extract numbers and alphanumeric sequences
_ALPHANUMERIC_PATTERN = re.compile(r'[a-zA-Z0-9]+')


def _normalize_keywords(keywords: list[str]) -> list[str]:
    """
    Normalize keywords by extracting alphanumeric portions.

    For example:
    - "2025年" -> ["2025年", "2025"]
    - "meta-data" -> ["meta-data", "meta", "data"]
    - "生活" -> ["生活"]

    Returns expanded list including both original and normalized variants.
    """
    normalized = set()

    for keyword in keywords:
        # Always include the original keyword
        normalized.add(keyword)

        # Extract alphanumeric sequences (numbers, English words)
        alphanumeric_parts = _ALPHANUMERIC_PATTERN.findall(keyword)
        for part in alphanumeric_parts:
            if len(part) >= 2:  # Only add if meaningful (2+ chars)
                normalized.add(part)

    return list(normalized)


def _contains_cjk(s: str) -> bool:
    """Check if string contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(_CJK_PATTERN.search(s))


def search_vector(
    session: Session,
    query_embedding: list[float],
    limit: int = 10,
) -> list[tuple[Document, float]]:
    """
    Vector similarity search using cosine distance.
    Returns list of (Document, similarity_score) tuples.
    """
    # Use cosine distance (1 - cosine_similarity), so lower is better
    # Convert to similarity score (higher is better) by using 1 - distance
    # Convert embedding list to PostgreSQL array format
    embedding_str = f"[{','.join(map(str, query_embedding))}]"

    search_query = text("""
        SELECT
            d.*,
            1 - (d.embedding <=> CAST(:embedding AS vector)) as similarity
        FROM documents d
        WHERE d.embedding IS NOT NULL
        ORDER BY d.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    results = session.execute(
        search_query,
        {"embedding": embedding_str, "limit": limit}
    ).fetchall()

    docs = []
    for row in results:
        doc = session.query(Document).filter(Document.id == row.id).first()
        if doc:
            docs.append((doc, row.similarity))
    return docs


def search_keyword(
    session: Session,
    query: str,
    limit: int = 10,
) -> list[tuple[Document, float]]:
    """
    Full-text keyword search using PostgreSQL tsvector.
    Supports universal language search.
    Returns list of (Document, rank_score) tuples.
    """
    if _contains_cjk(query):
        # Use ILIKE with pg_trgm GIN index for CJK (exact substring matching)
        search_query = text("""
            SELECT
                d.*,
                similarity(d.content_text, :query) as rank
            FROM documents d
            WHERE d.content_text ILIKE '%' || :query || '%'
            ORDER BY rank DESC
            LIMIT :limit
        """)
        params = {"query": query, "limit": limit}
    else:
        # For space-delimited languages, use 'simple' config (universal tokenization)
        search_query = text("""
            SELECT
                d.*,
                ts_rank_cd(d.content_tsvector, plainto_tsquery('simple', :query)) as rank
            FROM documents d
            WHERE d.content_tsvector @@ plainto_tsquery('simple', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)
        params = {"query": query, "limit": limit}

    results = session.execute(search_query, params).fetchall()

    docs = []
    for row in results:
        doc = session.query(Document).filter(Document.id == row.id).first()
        if doc:
            docs.append((doc, row.rank))
    return docs


def find_direct_matches(
    session: Session,
    keywords: list[str],
    limit: int = 10,
) -> list[tuple[Document, float]]:
    """
    Find documents with direct filename/title matches.

    A document is a direct match if its filename or title:
    1. Contains 'meta' AND any of the keywords, OR
    2. Exactly matches any keyword

    Keywords are normalized to extract alphanumeric parts (e.g., "2025年" -> "2025").

    Returns list of (Document, priority_score) tuples with high scores.
    Direct matches get priority score of 100.0 to ensure top ranking.
    """
    if not keywords:
        return []

    # Normalize keywords to include alphanumeric variants
    normalized_keywords = _normalize_keywords(keywords)

    # Build conditions for each keyword
    conditions = []
    params = {"limit": limit}
    param_counter = 0

    for keyword in normalized_keywords:
        # Use counter to avoid parameter name conflicts
        exact_param = f"exact_{param_counter}"
        pattern_param = f"pattern_{param_counter}"

        # Exact match: filename or title equals keyword (case-insensitive)
        conditions.append(f"(d.filename ILIKE :{exact_param} OR d.title ILIKE :{exact_param})")
        # Meta + keyword: contains both 'meta' and the keyword
        conditions.append(
            f"((d.filename ILIKE '%meta%' AND d.filename ILIKE :{pattern_param}) OR "
            f"(d.title ILIKE '%meta%' AND d.title ILIKE :{pattern_param}))"
        )

        params[exact_param] = keyword
        params[pattern_param] = f"%{keyword}%"
        param_counter += 1

    where_clause = " OR ".join(conditions)

    search_query = text(f"""
        SELECT d.*, 100.0 as priority
        FROM documents d
        WHERE {where_clause}
        ORDER BY d.filename
        LIMIT :limit
    """)

    results = session.execute(search_query, params).fetchall()

    docs = []
    for row in results:
        doc = session.query(Document).filter(Document.id == row.id).first()
        if doc:
            docs.append((doc, row.priority))

    return docs
