"""
Note data ingestion service.

Ingests note data from object storage (S3) or local filesystem (dev).
Uses content hashing for efficient change detection.
"""
import hashlib
import os
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import text

from config.database import get_db
from config.embedding_factory import EmbeddingConfig, create_embedding_client
from config.object_storage import list_objects, get_object
from entity.document import Document


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using configured provider."""
    if not texts:
        return []
    config = EmbeddingConfig(
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "openrouter"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY"),
    )
    client = create_embedding_client(config)
    response = client.embeddings.create(
        model=config.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


@dataclass
class SyncStats:
    """Statistics for sync operation."""
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    skipped: int = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


def _compute_hash(content: str) -> str:
    """Compute MD5 hash of content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _extract_title(content: str, filename: str | None = None) -> str | None:
    """Extract title from first heading line, fallback to filename."""
    lines = content.strip().split("\n")
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip()
    # Fallback to filename without extension
    if filename:
        return filename.removesuffix(".md")
    return None


def ingest_notes(prefix: str = "notes/", delete_missing: bool = True) -> dict:
    """
    Ingest note data from object storage with efficient change detection.

    Algorithm:
    1. Load existing documents (id -> content_hash) from DB
    2. List files from storage and compute hashes
    3. Compare to find: new, updated, deleted, unchanged
    4. Only process changed files

    Args:
        prefix: Object storage prefix to scan for note files (default: "notes/")
        delete_missing: If True, delete documents not found in storage (default: True)

    Returns:
        dict with status and detailed sync statistics
    """
    logger.info(f"Starting note sync from prefix: {prefix}")
    stats = SyncStats()

    # List all objects under the prefix
    keys = list_objects(prefix=prefix)
    storage_files = {k for k in keys if k.endswith(".md")}
    stats.skipped = len(keys) - len(storage_files)
    logger.info(f"Found {len(storage_files)} markdown files ({stats.skipped} skipped)")

    with get_db() as session:
        # Step 1: Load existing documents into memory (id -> content_hash)
        # Note: CHAR(36) pads with spaces, so strip() is needed for comparison
        existing_docs = {
            doc.id.strip(): doc.content_hash
            for doc in session.query(Document.id, Document.content_hash)
            .filter(Document.doc_type == 1)  # notes only
            .all()
        }
        logger.info(f"Found {len(existing_docs)} existing notes in DB")

        # Step 2: Build map of storage files (doc_id -> key)
        storage_map = {}
        for key in storage_files:
            doc_id = key.split("/")[-1]
            storage_map[doc_id] = key

        # Step 3: Categorize changes
        existing_ids = set(existing_docs.keys())
        storage_ids = set(storage_map.keys())

        to_create = storage_ids - existing_ids
        to_check = storage_ids & existing_ids  # May need update
        to_delete = existing_ids - storage_ids if delete_missing else set()

        logger.info(
            f"Changes: {len(to_create)} new, {len(to_check)} to check, "
            f"{len(to_delete)} to delete"
        )

        # Step 4: Process new files
        for doc_id in to_create:
            key = storage_map[doc_id]
            try:
                content = get_object(key, decode=True, parse_json=False)
                if content is None:
                    stats.errors.append({"key": key, "error": "Could not read"})
                    continue

                content_hash = _compute_hash(content)
                doc = Document(
                    id=doc_id,
                    filename=doc_id,
                    s3_key=key,
                    doc_type=1,
                    content_hash=content_hash,
                    title=_extract_title(content, doc_id),
                    content_text=content[:16384],
                )
                session.add(doc)
                stats.created += 1
                logger.debug(f"Created: {doc_id}")

            except Exception as e:
                logger.error(f"Error creating {key}: {e}")
                stats.errors.append({"key": key, "error": str(e)})

        # Step 5: Check existing files for updates
        for doc_id in to_check:
            key = storage_map[doc_id]
            try:
                content = get_object(key, decode=True, parse_json=False)
                if content is None:
                    stats.errors.append({"key": key, "error": "Could not read"})
                    continue

                content_hash = _compute_hash(content)

                # Skip if unchanged
                if existing_docs[doc_id] == content_hash:
                    stats.unchanged += 1
                    continue

                # Update changed document
                doc = session.query(Document).filter(Document.id == doc_id).first()
                if doc:
                    doc.s3_key = key
                    doc.content_hash = content_hash
                    doc.title = _extract_title(content, doc_id)
                    doc.content_text = content[:16384]
                    stats.updated += 1
                    logger.debug(f"Updated: {doc_id}")

            except Exception as e:
                logger.error(f"Error updating {key}: {e}")
                stats.errors.append({"key": key, "error": str(e)})

        # Step 6: Delete removed files
        if to_delete:
            session.query(Document).filter(Document.id.in_(to_delete)).delete(
                synchronize_session=False
            )
            stats.deleted = len(to_delete)
            logger.info(f"Deleted {stats.deleted} notes")

        # Step 7: Fix NULL titles using filename as fallback
        docs_without_title = (
            session.query(Document)
            .filter(Document.title.is_(None))
            .all()
        )
        if docs_without_title:
            for doc in docs_without_title:
                doc.title = doc.id.removesuffix(".md")
            logger.info(f"Fixed {len(docs_without_title)} NULL titles")

        # Step 8: Update tsvectors for all documents with NULL tsvector (must run after title fix)
        logger.info("Updating tsvectors...")
        result = session.execute(text("""
            UPDATE documents
            SET content_tsvector = to_tsvector('english', coalesce(title, '') || ' ' || content_text)
            WHERE content_tsvector IS NULL
        """))
        tsvector_updated = result.rowcount
        if tsvector_updated:
            logger.info(f"Updated {tsvector_updated} tsvectors")

        # Step 9: Generate embeddings for documents with NULL embedding
        docs_needing_embedding = (
            session.query(Document)
            .filter(Document.embedding.is_(None))
            .all()
        )
        if docs_needing_embedding:
            logger.info(f"Generating embeddings for {len(docs_needing_embedding)} documents...")
            # Process in batches of 100 (OpenAI limit is 2048)
            batch_size = 100
            for i in range(0, len(docs_needing_embedding), batch_size):
                batch = docs_needing_embedding[i:i + batch_size]
                texts = [
                    f"{doc.title or ''} {doc.content_text}".strip()
                    for doc in batch
                ]
                try:
                    embeddings = _get_embeddings(texts)
                    for doc, embedding in zip(batch, embeddings):
                        doc.embedding = embedding
                    logger.info(f"Generated embeddings for batch {i // batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    stats.errors.append({"batch": i // batch_size, "error": str(e)})

        session.commit()

    logger.info(
        f"Sync complete: {stats.created} created, {stats.updated} updated, "
        f"{stats.deleted} deleted, {stats.unchanged} unchanged"
    )

    return {
        "status": "success" if not stats.errors else "partial",
        "message": f"Synced notes: +{stats.created} ~{stats.updated} -{stats.deleted}",
        "created": stats.created,
        "updated": stats.updated,
        "deleted": stats.deleted,
        "unchanged": stats.unchanged,
        "skipped": stats.skipped,
        "errors": stats.errors if stats.errors else None,
    }


def handle_ingest_notes(message: dict) -> dict:
    """
    Handle ingest_notes action from worker.

    Args:
        message: Message dict with optional fields:
            - prefix: Storage prefix (default: "notes/")
            - delete_missing: Whether to delete missing docs (default: True)

    Returns:
        Sync result dict
    """
    prefix = message.get("prefix", "notes/")
    delete_missing = message.get("delete_missing", True)
    return ingest_notes(prefix=prefix, delete_missing=delete_missing)
