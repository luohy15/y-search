# TradingAgents/graph/embedding_factory.py

from typing import Optional
from openai import OpenAI
from pydantic import BaseModel, Field

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models.

    This model defines configuration for text embeddings used in memory and retrieval.
    """
    embedding_provider: str = Field(default="openai", description="Embedding provider name")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model name")
    embedding_base_url: Optional[str] = Field(
        default="https://api.openai.com/v1",
        description="Backend URL for embedding API (defaults to LLM base URL if not specified)"
    )
    embedding_api_key: Optional[str] = Field(default=None, description="API key for embedding service")


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
