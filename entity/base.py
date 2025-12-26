from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def now():
    return datetime.now(timezone.utc)


class BaseEntity(Base):
    __abstract__ = True
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=now, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=now, onupdate=now, server_default=func.now())