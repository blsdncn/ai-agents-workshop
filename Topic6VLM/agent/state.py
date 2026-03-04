"""LangGraph state objects for Exercise 1.

Current policy:
- Append-only conversation history in Ollama message shape.
- Images are attached only to the user turn that uploads them.
- No automatic image resend: user re-attaches explicitly when desired.
- Per-turn limits: up to 3 images, resized to max 720 px edge.
"""

from __future__ import annotations

from typing import Any, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage


class AgentState(TypedDict):
    """Conversation + transient turn buffers for one session.

    TODO(packages):
    - `langgraph` with message-aware reducers (`add_messages`) when wiring graph updates.

    TODO(technique):
    - Keep `messages` as the source of truth for model context.
    - Keep only current-turn upload data in `pending_*` buffers.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    pending_user_text: str | None
    pending_image_paths: list[str]
    pending_image_b64: list[str]
    last_model_text: str | None
    error: str | None


class AppConfig(dict):
    """Runtime configuration container.

    TODO(tool):
    - Prefer environment variables for local config (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`).
    - Optionally use `.env` + `python-dotenv` for local development ergonomics.
    """


def initial_state() -> AgentState:
    """Create initial state for a new chat session.

    TODO(technique):
    - If you add persistence, seed from checkpoint store by thread/session ID.
    """

    return {
        "messages": [],
        "pending_user_text": None,
        "pending_image_paths": [],
        "pending_image_b64": [],
        "last_model_text": None,
        "error": None,
    }
