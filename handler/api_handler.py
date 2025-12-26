import json

from service.example import list_examples


def api_handler(event, _context):
    """Handle API requests."""
    http_method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")

    # Get query parameters - handle both API Gateway v1 and v2 formats
    query_params = event.get("queryStringParameters") or {}

    # HTTP API v2 may have multiValueQueryStringParameters for repeated params
    if not query_params and "multiValueQueryStringParameters" in event:
        # Convert multi-value to single value (take first value)
        multi_params = event.get("multiValueQueryStringParameters") or {}
        query_params = {k: v[0] if isinstance(v, list) and v else v for k, v in multi_params.items()}

    if path == "/examples" and http_method == "GET":
        limit = int(query_params.get("limit", 100))
        offset = int(query_params.get("offset", 0))
        examples = list_examples(limit=limit, offset=offset)
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"examples": examples, "limit": limit, "offset": offset}),
        }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "message": "Y Search API",
            "method": http_method,
            "path": path,
            "query_params": query_params,
        }),
    }


if __name__ == "__main__":
    test_event = {
        "requestContext": {"http": {"method": "GET"}},
        "rawPath": "/examples",
        "queryStringParameters": {"symbol": "AAPL"},
    }
    result = api_handler(test_event, None)
    print(json.dumps(result, indent=2))
