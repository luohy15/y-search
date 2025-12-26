from typing import Optional

from config.database import get_db
from entity.example import Example
from repository import example as example_repo


def create_example(name: str, content: Optional[str] = None) -> dict:
    """Create a new Example record."""
    with get_db() as session:
        example = example_repo.create(session, name=name, content=content)
        return _to_dict(example)


def get_example_by_id(id: int) -> Optional[dict]:
    """Get an Example by ID."""
    with get_db() as session:
        example = example_repo.get_by_id(session, id)
        return _to_dict(example) if example else None


def get_example_by_name(name: str) -> Optional[dict]:
    """Get an Example by name."""
    with get_db() as session:
        example = example_repo.get_by_name(session, name)
        return _to_dict(example) if example else None


def list_examples(limit: int = 100, offset: int = 0) -> list[dict]:
    """List all Examples with pagination."""
    with get_db() as session:
        examples = example_repo.get_all(session, limit=limit, offset=offset)
        return [_to_dict(e) for e in examples]


def update_example(id: int, name: Optional[str] = None, content: Optional[str] = None) -> Optional[dict]:
    """Update an Example by ID."""
    with get_db() as session:
        example = example_repo.update(session, id, name=name, content=content)
        return _to_dict(example) if example else None


def delete_example(id: int) -> bool:
    """Delete an Example by ID."""
    with get_db() as session:
        return example_repo.delete(session, id)


def _to_dict(example: Example) -> dict:
    """Convert Example entity to dictionary."""
    return {
        "id": example.id,
        "name": example.name,
        "content": example.content,
    }
