"""
Embedding client factory with environment-based configuration.

Environment Variables:
    EMBEDDING_PROVIDER: Provider name - 'openai' or 'openrouter' (default: 'openai')
    EMBEDDING_MODEL: Model name (default: 'text-embedding-3-small')
    EMBEDDING_BASE_URL: API base URL (default: provider-specific)
    EMBEDDING_API_KEY: API key for embedding service
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Module-level singleton
_embedding_client: Optional[OpenAI] = None
_embedding_model: Optional[str] = None


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    embedding_provider: str = Field(default="openrouter", description="Embedding provider name")
    embedding_model: str = Field(default="google/gemini-embedding-001", description="Embedding model name")
    embedding_base_url: Optional[str] = Field(default=None, description="Backend URL for embedding API")
    embedding_api_key: Optional[str] = Field(default=None, description="API key for embedding service")


def _get_config_from_env() -> EmbeddingConfig:
    """Create EmbeddingConfig from environment variables."""
    return EmbeddingConfig(
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openrouter"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001"),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY"),
    )


def get_embedding_client() -> OpenAI:
    """Get singleton embedding client configured from environment."""
    global _embedding_client
    if _embedding_client is None:
        config = _get_config_from_env()
        _embedding_client = create_embedding_client(config)
    return _embedding_client


def get_embedding_model() -> str:
    """Get embedding model name from environment."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001")
    return _embedding_model


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (1536 dimensions)
    """
    if not texts:
        return []
    client = get_embedding_client()
    model = get_embedding_model()
    response = client.embeddings.create(model=model, input=texts, dimensions=1536)
    return [item.embedding for item in response.data]


def create_embedding_client(config: EmbeddingConfig) -> OpenAI:
    """Create and configure OpenAI client for embeddings based on provider.

    Args:
        config: Embedding configuration containing provider settings

    Returns:
        Configured OpenAI client for embeddings

    Raises:
        ValueError: If unsupported embedding provider is specified
    """
    provider = config.embedding_provider.lower()

    if provider in ["openai", "openrouter"]:
        return _create_openai_embedding_client(config)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def _create_openai_embedding_client(config: EmbeddingConfig) -> OpenAI:
    """Create OpenAI/OpenRouter embedding client."""
    provider = config.embedding_provider.lower()

    # Set default URL based on provider
    if provider == "openrouter":
        default_url = "https://openrouter.ai/api/v1"
    else:
        default_url = "https://api.openai.com/v1"

    base_url = config.embedding_base_url or default_url

    # Build client kwargs
    client_kwargs = {"base_url": base_url}
    if config.embedding_api_key:
        client_kwargs["api_key"] = config.embedding_api_key

    return OpenAI(**client_kwargs)
