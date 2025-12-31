"""
Completion (chat) client factory with environment-based configuration.

Environment Variables:
    LLM_PROVIDER: Provider name - 'openai' or 'openrouter' (default: 'openrouter')
    LLM_MODEL: Model name (default: 'google/gemini-3-flash-preview')
    LLM_BASE_URL: API base URL (default: provider-specific)
    LLM_API_KEY: API key for completion service (falls back to EMBEDDING_API_KEY)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Module-level singleton
_llm_client: Optional[OpenAI] = None
_llm_model: Optional[str] = None


class CompletionConfig(BaseModel):
    """Configuration for completion models."""
    llm_provider: str = Field(default="openrouter", description="Completion provider name")
    llm_model: str = Field(default="google/gemini-3-flash-preview", description="Completion model name")
    llm_base_url: Optional[str] = Field(default=None, description="Backend URL for completion API")
    llm_api_key: Optional[str] = Field(default=None, description="API key for completion service")


def _get_config_from_env() -> CompletionConfig:
    """Create CompletionConfig from environment variables."""
    # Fallback to EMBEDDING_API_KEY if LLM_API_KEY not set
    api_key = os.getenv("LLM_API_KEY") or os.getenv("EMBEDDING_API_KEY")

    return CompletionConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openrouter"),
        llm_model=os.getenv("LLM_MODEL", "google/gemini-3-flash-preview"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=api_key,
    )


def get_llm_client() -> OpenAI:
    """Get singleton completion client configured from environment."""
    global _llm_client
    if _llm_client is None:
        config = _get_config_from_env()
        _llm_client = create_llm_client(config)
    return _llm_client


def get_llm_model() -> str:
    """Get completion model name from environment."""
    global _llm_model
    if _llm_model is None:
        _llm_model = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
    return _llm_model


def get_completion(
    messages: list[dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> str:
    """
    Generate a completion for a list of messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Optional maximum tokens for completion
        temperature: Optional temperature for completion
        **kwargs: Additional parameters to pass to the API

    Returns:
        The completion text

    Example:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        response = get_completion(messages)
    """
    client = get_llm_client()
    model = get_llm_model()

    # Build request parameters
    params = {
        "model": model,
        "messages": messages,
        **kwargs
    }

    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature

    response = client.chat.completions.create(**params)

    return response.choices[0].message.content


def create_llm_client(config: CompletionConfig) -> OpenAI:
    """Create and configure OpenAI client for completions based on provider.

    Args:
        config: Completion configuration containing provider settings

    Returns:
        Configured OpenAI client for completions

    Raises:
        ValueError: If unsupported completion provider is specified
    """
    provider = config.llm_provider.lower()

    if provider in ["openai", "openrouter"]:
        return _create_openai_llm_client(config)
    else:
        raise ValueError(f"Unsupported completion provider: {config.llm_provider}")


def _create_openai_llm_client(config: CompletionConfig) -> OpenAI:
    """Create OpenAI/OpenRouter completion client."""
    provider = config.llm_provider.lower()

    # Set default URL based on provider
    if provider == "openrouter":
        default_url = "https://openrouter.ai/api/v1"
    else:
        default_url = "https://api.openai.com/v1"

    base_url = config.llm_base_url or default_url

    # Build client kwargs
    client_kwargs = {"base_url": base_url}
    if config.llm_api_key:
        client_kwargs["api_key"] = config.llm_api_key

    return OpenAI(**client_kwargs)
