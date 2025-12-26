from typing import Optional
from sqlalchemy.orm import Session

from entity.example import Example


def create(session: Session, name: str, content: Optional[str] = None) -> Example:
    """Create a new Example record."""
    example = Example(name=name, content=content)
    session.add(example)
    session.flush()
    return example


def get_by_id(session: Session, id: int) -> Optional[Example]:
    """Get an Example by ID."""
    return session.query(Example).filter(Example.id == id).first()


def get_by_name(session: Session, name: str) -> Optional[Example]:
    """Get an Example by name."""
    return session.query(Example).filter(Example.name == name).first()


def get_all(session: Session, limit: int = 100, offset: int = 0) -> list[Example]:
    """Get all Examples with pagination."""
    return session.query(Example).offset(offset).limit(limit).all()


def update(session: Session, id: int, name: Optional[str] = None, content: Optional[str] = None) -> Optional[Example]:
    """Update an Example by ID."""
    example = get_by_id(session, id)
    if not example:
        return None
    if name is not None:
        example.name = name
    if content is not None:
        example.content = content
    session.flush()
    return example


def delete(session: Session, id: int) -> bool:
    """Delete an Example by ID. Returns True if deleted."""
    example = get_by_id(session, id)
    if not example:
        return False
    session.delete(example)
    session.flush()
    return True
