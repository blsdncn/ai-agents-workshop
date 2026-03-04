"""Thin wrapper for Ollama multimodal chat calls.

This file is intentionally a skeleton so you can implement request details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OllamaSettings:
    """Connection and model options for Ollama.

    TODO(packages):
    - `ollama` Python SDK for direct local inference calls.
    - Optional fallback: `requests` for custom HTTP handling/retries.
    """

    model: str = "llava"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 90.0
    max_images_per_turn: int = 3


def build_multimodal_message(
    user_text: str, image_b64_list: list[str]
) -> dict[str, Any]:
    """Build one user message payload in Ollama message shape.

    TODO(technique):
    - Always include `role` and `content`.
    - Include `images` only when `image_b64_list` is non-empty.
    - Keep this function transport-focused and stateless.
    """

    raise NotImplementedError("TODO: implement message payload construction")


def chat_with_llava(messages: list[dict[str, Any]], settings: OllamaSettings) -> str:
    """Execute one chat completion and return model text.
    Arguments:
        messages - LangChain

    TODO(tool):
    - Add structured logging of latency and prompt length.
    - Add basic retry policy for transient local server errors.
    """

    raise NotImplementedError("TODO: implement Ollama chat call")
