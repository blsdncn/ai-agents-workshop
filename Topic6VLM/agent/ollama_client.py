"""Thin wrapper for Ollama multimodal chat calls.

This file is intentionally a skeleton so you can implement request details.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, reduce
import os
from typing import Any
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
)

from langchain_ollama import ChatOllama


@dataclass(slots=True)
class OllamaSettings:
    """Connection and model options for Ollama."""

    model: str = "llava"
    base_url: str = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    timeout_seconds: float = 90.0


@lru_cache
def _get_chatollama_cached(
    model: str, base_url: str, timeout_seconds: float
) -> ChatOllama:
    return ChatOllama(
        model=model, base_url=base_url, client_kwargs={"timeout": timeout_seconds}
    )


def get_chatollama(s: OllamaSettings) -> ChatOllama:
    return _get_chatollama_cached(s.model, s.base_url, s.timeout_seconds)


def chat_with_llava(messages: list[AnyMessage], settings: OllamaSettings) -> AIMessage:
    """Execute one chat completion and return model text.
    Arguments:
        messages - LangChain

    TODO(tool):
    - Add structured logging of latency and prompt length.
    - Add basic retry policy for transient local server errors.
    """
    chat_ollama = get_chatollama(settings)
    return chat_ollama.invoke(messages)
