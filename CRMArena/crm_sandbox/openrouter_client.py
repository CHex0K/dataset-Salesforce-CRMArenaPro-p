import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
from openai import OpenAI

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key not found. Set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment."
        )
    return api_key


def _resolve_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)


@lru_cache(maxsize=1)
def get_openrouter_client() -> OpenAI:
    """Return a cached OpenAI client configured for OpenRouter."""
    return OpenAI(base_url=_resolve_base_url(), api_key=_resolve_api_key())


def chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """Helper that proxies chat completion calls through the OpenRouter OpenAI client."""
    client = get_openrouter_client()
    extra_body = extra_body or {}
    extra_body.setdefault("usage", {"include": True})
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
