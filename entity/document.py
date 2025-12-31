from sqlalchemy import Column, String, Index, SmallInteger, CHAR
from sqlalchemy.dialects.postgresql import TSVECTOR
from pgvector.sqlalchemy import Vector

from entity.base import BaseEntity


class Document(BaseEntity):
    """Document entity for storing indexed documents."""
    __tablename__ = 'documents'

    # Primary key - UUID
    id = Column(CHAR(36), primary_key=True)

    # File identification
    filename = Column(String(255), nullable=False, unique=True)  # Original filename
    s3_key = Column(String(512), nullable=False)  # S3 object key

    # Document type: 1=note, 2=link
    doc_type = Column(SmallInteger, nullable=False, default=1)

    # Content hash for change detection (MD5 hex digest)
    content_hash = Column(CHAR(32), nullable=True)

    # Content
    title = Column(String(512), nullable=True)  # Document title or filename
    content_text = Column(String(16384), nullable=False)  # Text content for display (max 16KB)

    # Search indexes
    content_tsvector = Column(TSVECTOR, nullable=True)  # PostgreSQL full-text search
    embedding = Column(Vector(1536), nullable=True)  # Reduced dimension for HNSW compatibility

    __table_args__ = (
        # GIN index for full-text search
        Index('idx_documents_content_tsvector', 'content_tsvector', postgresql_using='gin'),
        # GIN index with pg_trgm for similarity search (supports CJK)
        Index('idx_documents_content_trgm', 'content_text', postgresql_using='gin',
              postgresql_ops={'content_text': 'gin_trgm_ops'}),
        # HNSW index for vector similarity (faster than IVFFlat for < 1M rows)
        Index('idx_documents_embedding', 'embedding', postgresql_using='hnsw',
              postgresql_with={'m': 16, 'ef_construction': 64},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),
    )
