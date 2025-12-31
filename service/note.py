"""
Note data ingestion service.

Ingests note data from object storage (S3) or local filesystem (dev).
Uses content hashing for efficient change detection.
"""
import hashlib
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy import text

from config.database import get_db
from config.embedding_factory import get_embeddings
from config.object_storage import list_objects_with_metadata, get_object
from entity.document import Document


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


def ingest_notes(prefix: str = "", delete_missing: bool = True, count: int | None = None) -> dict:
    """
    Ingest note data from object storage with efficient change detection.

    Algorithm:
    1. Load existing documents (id -> content_hash) from DB
    2. List files from storage and compute hashes
    3. Compare to find: new, updated, deleted, unchanged
    4. Only process changed files

    Args:
        prefix: Object storage prefix to scan for note files (default: "")
        delete_missing: If True, delete documents not found in storage (default: True)
        count: Max number of new documents to create (default: None = unlimited)

    Returns:
        dict with status and detailed sync statistics
    """
    logger.info(f"Starting note sync from prefix: {prefix}")
    stats = SyncStats()

    # List all objects under the prefix with metadata
    objects_with_metadata = list_objects_with_metadata(prefix=prefix, max_keys=10000)
    md_files_with_metadata = [
        obj for obj in objects_with_metadata
        if obj['key'].endswith(".md") and not obj['key'].startswith("assets/")
    ]
    stats.skipped = len(objects_with_metadata) - len(md_files_with_metadata)
    logger.info(f"Found {len(md_files_with_metadata)} markdown files ({stats.skipped} skipped)")

    with get_db() as session:
        # Step 1: Load existing documents into memory (filename -> content_hash)
        existing_docs = {
            doc.filename: doc.content_hash
            for doc in session.query(Document.filename, Document.content_hash)
            .filter(Document.doc_type == 1)  # notes only
            .all()
        }
        logger.info(f"Found {len(existing_docs)} existing notes in DB")

        # Step 2: Build map of storage files (filename -> {key, last_modified})
        storage_map = {}
        for obj in md_files_with_metadata:
            key = obj['key']
            filename = key.split("/")[-1]
            storage_map[filename] = {
                'key': key,
                'last_modified': obj['last_modified']
            }

        # Step 3: Categorize changes
        existing_filenames = set(existing_docs.keys())
        storage_filenames = set(storage_map.keys())

        to_create = storage_filenames - existing_filenames
        to_check = storage_filenames & existing_filenames  # May need update
        to_delete = existing_filenames - storage_filenames if delete_missing else set()

        logger.info(
            f"Changes: {len(to_create)} new, {len(to_check)} to check, "
            f"{len(to_delete)} to delete"
        )

        # Step 4: Process new files (sorted by last_modified descending - newest first)
        to_create_sorted = sorted(
            to_create,
            key=lambda filename: storage_map[filename]['last_modified'],
            reverse=True
        )

        for filename in to_create_sorted:
            # Stop if we've reached the count limit
            if count is not None and stats.created >= count:
                logger.info(f"Reached count limit of {count} created documents")
                break

            key = storage_map[filename]['key']
            try:
                content = get_object(key, decode=True, parse_json=False)
                if content is None:
                    stats.errors.append({"key": key, "error": "Could not read"})
                    continue

                content_hash = _compute_hash(content)
                doc = Document(
                    id=str(uuid.uuid4()),
                    filename=filename,
                    s3_key=key,
                    doc_type=1,
                    content_hash=content_hash,
                    title=_extract_title(content, filename),
                    content_text=content[:16384],
                )
                session.add(doc)
                stats.created += 1
                logger.debug(f"Created: {filename}")

            except Exception as e:
                logger.error(f"Error creating {key}: {e}")
                stats.errors.append({"key": key, "error": str(e)})

        # (skip if count is specified)
        if count is None:
            # Step 5: Check existing files for updates
            for filename in to_check:
                key = storage_map[filename]['key']
                try:
                    content = get_object(key, decode=True, parse_json=False)
                    if content is None:
                        stats.errors.append({"key": key, "error": "Could not read"})
                        continue

                    content_hash = _compute_hash(content)

                    # Skip if unchanged
                    if existing_docs[filename] == content_hash:
                        stats.unchanged += 1
                        continue

                    # Update changed document
                    doc = session.query(Document).filter(Document.filename == filename).first()
                    if doc:
                        doc.s3_key = key
                        doc.content_hash = content_hash
                        doc.title = _extract_title(content, filename)
                        doc.content_text = content[:16384]
                        stats.updated += 1
                        logger.debug(f"Updated: {filename}")

                except Exception as e:
                    logger.error(f"Error updating {key}: {e}")
                    stats.errors.append({"key": key, "error": str(e)})

            # Step 6: Delete removed files
            if to_delete:
                session.query(Document).filter(Document.filename.in_(to_delete)).delete(
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
                doc.title = doc.filename.removesuffix(".md")
            logger.info(f"Fixed {len(docs_without_title)} NULL titles")

        # Step 8: Update tsvectors for all documents with NULL tsvector (must run after title fix)
        # Use 'simple' config for universal language support (CJK uses pg_bigm separately)
        logger.info("Updating tsvectors...")
        result = session.execute(text("""
            UPDATE documents
            SET content_tsvector = to_tsvector('simple', coalesce(title, '') || ' ' || content_text)
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
                    embeddings = get_embeddings(texts)
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


def fill_null_embeddings(batch_size: int = 100) -> dict:
    """
    Check for documents with NULL embeddings and fill them.

    Args:
        batch_size: Number of documents to process per batch (default: 100)

    Returns:
        dict with status and statistics
    """
    logger.info("Starting fill_null_embeddings task")
    success_count = 0
    error_count = 0
    errors = []

    with get_db() as session:
        # Find all documents with NULL embeddings
        docs_needing_embedding = (
            session.query(Document)
            .filter(Document.embedding.is_(None))
            .all()
        )

        total_docs = len(docs_needing_embedding)
        logger.info(f"Found {total_docs} documents with NULL embeddings")

        if not docs_needing_embedding:
            return {
                "status": "success",
                "message": "No documents need embeddings",
                "total": 0,
                "success": 0,
                "errors": 0,
            }

        # Process in batches
        for i in range(0, total_docs, batch_size):
            batch = docs_needing_embedding[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Prepare texts, filtering out empty ones
            texts = []
            valid_docs = []
            for doc in batch:
                text = f"{doc.title or ''} {doc.content_text or ''}".strip()
                if text:
                    texts.append(text)
                    valid_docs.append(doc)
                else:
                    logger.warning(f"Skipping document {doc.id} - empty text")
                    error_count += 1
                    errors.append({
                        "doc_id": doc.id,
                        "error": "Empty text content"
                    })

            if not texts:
                logger.warning(f"Batch {batch_num}: No valid texts to process")
                continue

            try:
                logger.info(f"Processing batch {batch_num}/{(total_docs + batch_size - 1) // batch_size} ({len(texts)} docs)")
                embeddings = get_embeddings(texts)

                # Validate we got embeddings back
                if not embeddings:
                    raise ValueError("No embedding data received")

                if len(embeddings) != len(texts):
                    raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")

                # Assign embeddings to documents
                for doc, embedding in zip(valid_docs, embeddings):
                    if embedding and len(embedding) > 0:
                        doc.embedding = embedding
                        success_count += 1
                    else:
                        logger.warning(f"Empty embedding for document {doc.id}")
                        error_count += 1
                        errors.append({
                            "doc_id": doc.id,
                            "error": "Empty embedding vector"
                        })

                session.commit()
                logger.info(f"Batch {batch_num} complete: {len(embeddings)} embeddings generated")

            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                error_count += len(texts)
                errors.append({
                    "batch": batch_num,
                    "count": len(texts),
                    "error": str(e)
                })
                session.rollback()

    logger.info(f"Fill embeddings complete: {success_count} success, {error_count} errors")

    return {
        "status": "success" if error_count == 0 else "partial",
        "message": f"Processed {total_docs} documents: {success_count} success, {error_count} errors",
        "total": total_docs,
        "success": success_count,
        "errors": error_count,
        "error_details": errors if errors else None,
    }


def handle_ingest_notes(message: dict) -> dict:
    """
    Handle ingest_notes action from worker.

    Args:
        message: Message dict with optional fields:
            - prefix: Storage prefix (default: "")
            - delete_missing: Whether to delete missing docs (default: True)
            - count: Max number of new documents to create (default: None = unlimited)

    Returns:
        Sync result dict
    """
    prefix = message.get("prefix", "")
    delete_missing = message.get("delete_missing", True)
    count = message.get("count")
    return ingest_notes(prefix=prefix, delete_missing=delete_missing, count=count)


def handle_fill_embeddings(message: dict) -> dict:
    """
    Handle fill_embeddings action from worker.

    Args:
        message: Message dict with optional fields:
            - batch_size: Number of documents per batch (default: 100)

    Returns:
        Fill embeddings result dict
    """
    batch_size = message.get("batch_size", 100)
    return fill_null_embeddings(batch_size=batch_size)
