"""
Base database configuration with automatic dialect detection.

Supports both PostgreSQL (production) and SQLite (local development) based on
the DATABASE_URL environment variable.

Examples:
    PostgreSQL: postgresql+psycopg://user:pass@host:port/dbname
    SQLite: sqlite:///path/to/database.db
    SQLite (in-memory): sqlite:///:memory:
"""

from __future__ import annotations
import os
from typing import Generator, Optional
from contextlib import contextmanager
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger

# Load environment variables
load_dotenv()

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None

def _get_database_url() -> str:
    """
    Get database URL from environment variables.

    Defaults to SQLite in-memory database if DATABASE_URL is not set.

    Returns:
        Database connection URL
    """
    database_url = os.getenv("DATABASE_URL_DEV", os.getenv("DATABASE_URL"))
    if database_url:
        return database_url

    # Default to SQLite in-memory
    logger.info("No DATABASE_URL found, using SQLite in-memory database")
    return "sqlite:///:memory:"


def _is_postgresql(url: str) -> bool:
    """Check if the database URL is for PostgreSQL"""
    return url.startswith("postgresql://") or url.startswith("postgresql+psycopg://")


def _is_sqlite(url: str) -> bool:
    """Check if the database URL is for SQLite"""
    return url.startswith("sqlite:///")


def _get_connect_args(database_url: str) -> dict:
    """
    Get database-specific connection arguments.

    Args:
        database_url: Database connection URL

    Returns:
        Dictionary of connection arguments
    """
    if _is_sqlite(database_url):
        # SQLite connection arguments
        return {
            "check_same_thread": False,  # Allow multi-threaded access
        }

    return {}


def _get_engine_kwargs(database_url: str) -> dict:
    """
    Get database-specific engine configuration.

    Args:
        database_url: Database connection URL

    Returns:
        Dictionary of engine kwargs
    """
    connect_args = _get_connect_args(database_url)

    if _is_postgresql(database_url):
        # PostgreSQL pool settings
        return {
            "pool_pre_ping": True,
            "pool_size": 20,  # Increased from default 5 for higher concurrency
            "max_overflow": 40,  # Increased from default 10 for peak load handling
            "pool_recycle": 3600,  # Recycle connections after 1 hour to prevent stale connections
            "pool_timeout": 5,
            "echo": os.environ.get("DB_ECHO", "false").lower() == "true",
            "connect_args": connect_args,
        }
    elif _is_sqlite(database_url):
        # SQLite settings (no connection pooling needed)
        return {
            "connect_args": connect_args,
        }

    # Default settings
    return {
        "connect_args": connect_args,
    }


def init_db() -> None:
    """
    Initialize the database engine and session factory.

    This function is called automatically on import, but can be called
    manually to reinitialize the connection.
    """
    global _engine, _SessionLocal

    if _engine is not None:
        return

    database_url = _get_database_url()
    engine_kwargs = _get_engine_kwargs(database_url)

    logger.info(f"Initializing database connection: {database_url.split('@')[-1] if '@' in database_url else database_url}")

    _engine = create_engine(database_url, **engine_kwargs)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    # Verify connection
    if _is_postgresql(database_url):
        from sqlalchemy import text
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection verified")
    else:
        logger.info("SQLite database initialized")


def get_engine() -> Engine:
    """
    Get the SQLAlchemy engine instance.

    Returns:
        SQLAlchemy Engine instance
    """
    if _engine is None:
        init_db()
    return _engine  # type: ignore


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic commit/rollback.

    Usage:
        with get_db() as session:
            # Use session here
            session.add(entity)
            # Automatic commit on success, rollback on exception

    Yields:
        SQLAlchemy Session instance
    """
    if _SessionLocal is None:
        init_db()

    session: Session = _SessionLocal()  # type: ignore
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_session() -> Session:
    """
    Get a database session with manual lifecycle management.

    Note: You must call session.close() when done.
    Consider using get_db() context manager instead.

    Returns:
        SQLAlchemy Session instance
    """
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()  # type: ignore


def init_tables() -> None:
    """
    Create all database tables defined in the entity models.
    """
    if _engine is None:
        init_db()

    from entity import Base  # Import here to load all entity models

    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=_engine)
    logger.info("Database tables created successfully")
