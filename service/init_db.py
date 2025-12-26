"""
Database initialization service.

Creates PostgreSQL extensions and tables for search functionality.
"""
from loguru import logger
from sqlalchemy import text

from config.database import get_db
from entity import Base
from entity.document import Document  # noqa: F401 - needed for table creation


def init_db() -> dict:
    """
    Initialize the database.

    1. Creates pgvector extension
    2. Creates pg_trgm extension (for fuzzy matching)
    3. Creates the documents table with indexes
    """
    with get_db() as session:
        # Create extensions
        logger.info("Creating PostgreSQL extensions...")

        try:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("Created pgvector extension")
        except Exception as e:
            logger.warning(f"Could not create vector extension: {e}")

        try:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            logger.info("Created pg_trgm extension")
        except Exception as e:
            logger.warning(f"Could not create pg_trgm extension: {e}")

        session.commit()

        # Create tables
        logger.info("Creating documents table...")
        Base.metadata.create_all(session.get_bind())

        logger.info("Database initialized successfully")

        return {
            "status": "success",
            "message": "Database initialized",
            "tables": ["documents"],
            "extensions": ["vector", "pg_trgm"],
        }


def handle_init_db() -> dict:
    """Handle init_db action from worker."""
    return init_db()


def handler(event: dict, context) -> dict:
    """Lambda handler for database initialization."""
    return init_db()
