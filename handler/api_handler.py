"""
Lambda handler for search API.
"""
import json
from typing import Any

from loguru import logger

from service import document as document_service


def handler(event: dict, _context: Any) -> dict:
    """
    Main Lambda handler for search API.

    Routes:
        GET /search - Search documents
    """
    logger.info(f"Received event: {json.dumps(event)}")

    # Extract HTTP method and path
    http_method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")

    # Get query parameters - handle both API Gateway v1 and v2 formats
    query_params = event.get("queryStringParameters") or {}

    # HTTP API v2 may have multiValueQueryStringParameters for repeated params
    if not query_params and "multiValueQueryStringParameters" in event:
        # Convert multi-value to single value (take first value)
        multi_params = event.get("multiValueQueryStringParameters") or {}
        query_params = {k: v[0] if isinstance(v, list) and v else v for k, v in multi_params.items()}

    # Route request
    try:
        if path == "/search" and http_method == "GET":
            return _handle_search(query_params)
        else:
            return _response(404, {"error": "Not found"})
    except Exception as e:
        logger.exception(f"Error handling request: {e}")
        return _response(500, {"error": str(e)})


def _handle_search(query_params: dict) -> dict:
    """
    Handle GET /search

    Query parameters:
        query: Search query string (required)
        limit: Maximum number of results to search (optional, default 10)
        min_score: Minimum relevance score threshold (optional, default 0.015)
        max_sources: Maximum documents for synthesis (optional, default 10)
    """
    query = query_params.get("query")
    if not query:
        return _response(400, {"error": "Missing required parameter: query"})

    limit = int(query_params.get("limit", 10))
    min_score = float(query_params.get("min_score", 0.015))
    max_sources = int(query_params.get("max_sources", 10))

    result = document_service.search_with_synthesis(
        query=query, limit=limit, min_score=min_score, max_sources=max_sources
    )

    return _response(200, result)


def _response(status_code: int, body: dict) -> dict:
    """Build API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


if __name__ == "__main__":
    test_event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/search",
        "queryStringParameters": {"query": "我是谁", "limit": "5"},
    }
    result = handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2, ensure_ascii=False))