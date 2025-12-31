"""
Document search service.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from loguru import logger

from config.database import get_db
from config.embedding_factory import get_embeddings
from config.llm_factory import get_completion
from config.object_storage import get_object
from repository import document as document_repo

# RAG Synthesis Constants
MIN_RELEVANCE_SCORE = 0.015  # Top 3-7 documents typically
MAX_SOURCE_DOCUMENTS = 10
MAX_CONTENT_CHARS_PER_DOC = 8000  # ~2000 tokens

# Keyword Extraction Prompt
KEYWORD_EXTRACTION_PROMPT = """Extract 3-5 search keywords from the user's natural language query.

Guidelines:
- Extract key concepts, entities, and important terms
- For questions, extract the core topic words (not question words like "what", "how", "is")
- Keep important phrases intact (e.g., "machine learning" not separate "machine" "learning")
- Preserve original language (don't translate)
- Focus on words that would appear in relevant documents

User Query: {query}

Return ONLY a JSON array of keyword strings, nothing else.
Example: ["keyword1", "keyword2", "keyword3"]"""

# LLM Prompts
SYNTHESIS_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided documents.

Your task:
1. Analyze the user's question and the provided source documents
2. Generate a comprehensive, accurate answer using ONLY information from the sources
3. Include inline citations [1], [2], etc. referencing specific documents
4. If the sources don't contain enough information, acknowledge the limitation
5. Maintain the same language as the user's question (support English, Chinese, etc.)

Guidelines:
- Be concise but thorough (2-4 paragraphs)
- Use clear, professional language
- Cite sources for all factual claims
- Don't speculate beyond the provided information
- If sources conflict, present multiple perspectives with citations"""

SYNTHESIS_USER_PROMPT = """Question: {query}

Source Documents:

{documents}

Please provide a comprehensive answer to the question based on these sources, including inline citations [1], [2], etc."""


def _extract_keywords(query: str) -> list[str]:
    """
    Extract search keywords from natural language query using LLM.

    Args:
        query: Natural language query string

    Returns:
        List of 3-5 extracted keywords
    """
    try:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(query=query)
        response = get_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        )

        # Parse JSON response
        keywords = json.loads(response.strip())

        # Validate it's a list of strings
        if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
            logger.warning(f"Invalid keyword extraction response: {response}")
            return [query]  # Fallback to original query

        logger.info(f"Extracted keywords from '{query}': {keywords}")
        return keywords

    except Exception as e:
        logger.error(f"Keyword extraction failed: {e}")
        return [query]  # Fallback to original query


def _select_documents_for_context(
    search_results: list[tuple],
    min_score: float = MIN_RELEVANCE_SCORE,
    max_docs: int = MAX_SOURCE_DOCUMENTS,
) -> list[tuple]:
    """
    Select high-relevance documents for LLM context.

    Args:
        search_results: List of (Document, RRF_score) tuples
        min_score: Minimum RRF score threshold
        max_docs: Maximum documents to include

    Returns:
        List of (Document, score) tuples above threshold (up to max_docs)
    """
    filtered = [(doc, score) for doc, score in search_results if score >= min_score]
    return filtered[:max_docs]


def _fetch_full_content(doc) -> str:
    """
    Fetch full document content from S3, with fallback to truncated version.

    Args:
        doc: Document entity with s3_key

    Returns:
        Full content string (or truncated if fetch fails)
    """
    try:
        full_content = get_object(doc.s3_key, decode=True, parse_json=False)
        if full_content:
            return full_content
        logger.warning(f"Failed to fetch {doc.s3_key}, using truncated content")
    except Exception as e:
        logger.error(f"Error fetching {doc.s3_key}: {e}")

    # Fallback to truncated content
    return doc.content_text


def _fetch_multiple_contents(docs: list) -> dict[str, str]:
    """
    Fetch content for multiple documents in parallel.

    Args:
        docs: List of Document entities

    Returns:
        Map of doc_id -> content
    """
    content_map = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_full_content, doc): doc.id for doc in docs}

        for future in as_completed(futures, timeout=30):
            doc_id = futures[future]
            try:
                content_map[doc_id] = future.result(timeout=5)
            except Exception as e:
                logger.error(f"Error fetching content for {doc_id}: {e}")
                # Fallback handled in _fetch_full_content

    return content_map


def _format_documents_for_llm(docs: list, contents: dict[str, str]) -> str:
    """
    Format documents as numbered context for LLM.

    Args:
        docs: List of Document entities (in relevance order)
        contents: Map of doc_id -> full_content

    Returns:
        Formatted string with numbered documents
    """
    formatted_docs = []

    for i, doc in enumerate(docs, 1):
        content = contents.get(doc.id, doc.content_text)

        # Truncate to prevent token overflow
        truncated_content = content[:MAX_CONTENT_CHARS_PER_DOC]
        if len(content) > MAX_CONTENT_CHARS_PER_DOC:
            truncated_content += "...[truncated]"

        formatted_docs.append(f"""[{i}] Title: {doc.title}
Filename: {doc.filename}

{truncated_content}
""")

    return "\n\n".join(formatted_docs)


def _synthesize_answer(query: str, docs: list, scores: dict[str, float]) -> dict:
    """
    Generate LLM-synthesized answer with citations.

    Args:
        query: User's search query
        docs: Relevant documents (in order)
        scores: Map of doc_id -> RRF score

    Returns:
        dict with answer, sources, and metadata
    """
    from config.llm_factory import get_llm_client, get_llm_model

    # 1. Fetch full content in parallel
    contents = _fetch_multiple_contents(docs)

    # 2. Format documents for LLM
    formatted_docs = _format_documents_for_llm(docs, contents)

    # 3. Build messages
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": SYNTHESIS_USER_PROMPT.format(query=query, documents=formatted_docs)},
    ]

    # 4. Call LLM
    client = get_llm_client()
    model = get_llm_model()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.3,
        )
        answer = response.choices[0].message.content

        # 5. Build source citations
        sources = [
            {
                "index": i + 1,
                "document_id": doc.id,
                "filename": doc.filename,
                "title": doc.title,
                "relevance_score": round(float(scores.get(doc.id, 0.0)), 4),
            }
            for i, doc in enumerate(docs)
        ]

        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "model": model,
                "documents_used": len(docs),
                "total_tokens": response.usage.total_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        }

    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        raise


def _reciprocal_rank_fusion(
    keyword_results: list[tuple],
    vector_results: list[tuple],
    k: int = 60,
) -> list[tuple]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1/(k + rank_i)
    where k is a constant (typically 60) that prevents high rankings from dominating.

    Args:
        keyword_results: List of (doc, score) from keyword search
        vector_results: List of (doc, score) from vector search
        k: Ranking constant (default 60, per original RRF paper)

    Returns:
        List of (doc, rrf_score) sorted by combined score descending
    """
    rrf_scores = {}
    doc_map = {}

    # Score from keyword search
    for rank, (doc, _) in enumerate(keyword_results, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1 / (k + rank)
        doc_map[doc.id] = doc

    # Score from vector search
    for rank, (doc, _) in enumerate(vector_results, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0) + 1 / (k + rank)
        doc_map[doc.id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids]


def search_with_synthesis(
    query: str,
    limit: int = 10,
    min_score: float = MIN_RELEVANCE_SCORE,
    max_sources: int = MAX_SOURCE_DOCUMENTS,
) -> dict:
    """
    Hybrid search with LLM-synthesized answer.

    Combines keyword, vector, and metadata search using RRF, then uses LLM to generate
    a comprehensive answer from the most relevant documents.

    Process:
    1. Extract keywords from natural language query using LLM
    2. Find direct matches (filename contains 'meta' + keyword, OR exact match) - bypasses RRF
    3. Run keyword search for EACH extracted keyword (searches content)
    4. Run vector search with original query (works well with natural language)
    5. Combine content/vector results using Reciprocal Rank Fusion (RRF)
    6. Place direct matches at top, followed by RRF-ranked results
    7. Synthesize answer from top documents

    Args:
        query: Search query string (can be natural language)
        limit: Maximum number of documents to search
        min_score: Minimum RRF score threshold for inclusion
        max_sources: Maximum documents to use for synthesis

    Returns:
        dict with query, answer, sources, and metadata
    """
    logger.info(f"Hybrid search with synthesis: '{query}' (limit={limit})")

    # 1. Extract keywords from natural language query
    keywords = _extract_keywords(query)

    # 2. Get query embedding for vector search (use original query)
    query_embedding = get_embeddings([query])[0]

    with get_db() as session:
        # Fetch more results from each method to improve RRF quality
        fetch_limit = limit * 2

        # 3. Find direct matches (meta+keyword or exact match) - these bypass RRF
        # Keywords are normalized internally (e.g., "2025年" -> ["2025年", "2025"])
        direct_matches = document_repo.find_direct_matches(session, keywords, limit=fetch_limit)
        direct_match_ids = {doc.id for doc, _ in direct_matches}
        if direct_matches:
            logger.info(f"Found {len(direct_matches)} direct matches: {[doc.filename for doc, _ in direct_matches]}")
        else:
            logger.debug("No direct matches found")

        # 4. Run keyword search for EACH extracted keyword
        all_keyword_results = []
        for keyword in keywords:
            keyword_results = document_repo.search_keyword(session, keyword, limit=fetch_limit)
            all_keyword_results.extend(keyword_results)
            logger.debug(f"Keyword '{keyword}': {len(keyword_results)} results")

        # 5. Run vector search with original natural language query
        vector_results = document_repo.search_vector(session, query_embedding, limit=fetch_limit)

        logger.debug(
            f"Total keyword results: {len(all_keyword_results)}, "
            f"Vector results: {len(vector_results)}"
        )

        # 7. Combine using RRF (automatically deduplicates)
        # Filter out documents already in direct_matches to avoid duplicates
        filtered_keyword = [(doc, score) for doc, score in all_keyword_results if doc.id not in direct_match_ids]
        filtered_vector = [(doc, score) for doc, score in vector_results if doc.id not in direct_match_ids]

        rrf_results = _reciprocal_rank_fusion(
            filtered_keyword, filtered_vector
        )

        # 8. Combine: direct matches first (high priority), then RRF results
        combined_results = direct_matches + rrf_results

        # Select high-relevance documents for synthesis
        selected = _select_documents_for_context(
            combined_results, min_score=min_score, max_docs=max_sources
        )

        if not selected:
            # No documents meet threshold
            return {
                "query": query,
                "answer": "No relevant documents found for your query. Please try different keywords or rephrasing your question.",
                "sources": [],
                "metadata": {
                    "documents_used": 0,
                    "search_results_total": len(combined_results),
                    "extracted_keywords": keywords,
                },
            }

        # Extract documents and scores
        docs = [doc for doc, _ in selected]
        scores = {doc.id: score for doc, score in selected}

        # Synthesize answer
        result = _synthesize_answer(query, docs, scores)

        # Add query and search metadata
        result["query"] = query
        result["metadata"]["search_results_total"] = len(combined_results)
        result["metadata"]["extracted_keywords"] = keywords

        return result
