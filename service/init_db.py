from typing import Any, Dict

from loguru import logger

from config.database import init_db, init_tables


def handle_init_db() -> Dict[str, Any]:
    """
    Initialize database schema and populate reference tables

    Returns:
        Dict with initialization results including table counts
    """
    try:
        logger.info("Starting database initialization")

        # Initialize database connection
        logger.info("Initializing database connection")
        init_db()
        logger.info("Database connection initialized successfully")

        # Create tables if they don't exist
        logger.info("Creating database tables")
        init_tables()
        logger.info("Database tables created successfully")

        return {
            "status": "success",
            "message": "Database initialized successfully",
        }

    except Exception as e:
        logger.exception(f"Database initialization failed: {e}")
        return {
            "status": "error",
            "error": str(e),
        }
