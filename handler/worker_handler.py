import json
from datetime import datetime, timezone

from service.init_db import handle_init_db


def route_message(message):
    """
    Route message to appropriate handler based on 'action' field.
    """
    action = message.get("action")

    if action == "init_db":
        return handle_init_db()

    return {
        "action": action,
        "status": "processed",
        "message": message,
    }


def worker_handler(event, _context):
    """
    Universal message router for processing messages.
    Routes messages based on 'action' field to appropriate handlers.
    """
    try:
        if "Records" in event:
            results = []
            for record in event["Records"]:
                message_body = json.loads(record["body"])
                result = route_message(message_body)
                results.append(result)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": f"Processed {len(results)} messages",
                    "results": results,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
            }
        else:
            result = route_message(event)
            return result

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        }


if __name__ == "__main__":
    test_event = {"action": "init_db", "data": "sample"}
    result = worker_handler(test_event, None)
    print(json.dumps(result, indent=2))
