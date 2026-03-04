"""Gradio UI skeleton for Exercise 1 image chat.

This is intentionally scaffolding only.
"""

from __future__ import annotations

from typing import Any


def on_submit(
    user_text: str,
    image_paths: list[str] | None,
    chat_state: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Handle one user turn.

    TODO(tool):
    - Use Gradio `gr.Chatbot` + `gr.State`.
    - Use `gr.File(file_count="multiple", type="filepath")` for per-turn uploads.
    - Thread-safe session state if multiple users connect.

    TODO(technique):
    - Enforce max 3 images per turn at the UI layer before graph invocation.
    - Convert UI payload into graph input state (`pending_user_text`, `pending_image_paths`).
    - Invoke compiled graph and map assistant response back to chat transcript.
    """

    raise NotImplementedError("TODO: implement submit callback")


def build_app() -> Any:
    """Create and return Gradio Blocks app.

    TODO(packages):
    - `gradio` Blocks, Chatbot, Textbox, File, State, Button.

    TODO(technique):
    - Keep upload control available every turn for optional re-attach.
    - Label upload limits clearly: max 3 images/turn, resized to 720 px max edge.
    - Add a clear button that resets both UI and graph state.
    """

    raise NotImplementedError("TODO: implement Gradio app layout")
