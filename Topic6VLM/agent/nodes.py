"""LangGraph nodes for Exercise 1.

Best-practice startup notes from LangGraph docs:
- Keep nodes single-purpose: validate input, transform state, or call model.
- Return partial state updates from each node.
- Keep message history append-only unless running an explicit summary/trim strategy.


"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, content as msg_content

from .ollama_client import OllamaSettings, chat_with_llava
from .state import AgentState
from pathlib import Path
import cv2
import base64
from util.imgUtils import scale_to_max_dimension

from agent import ollama_client

MAX_IMAGES_PER_TURN = 3
MAX_IMAGE_EDGE_PX = 720
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def ingest_user_turn(state: AgentState) -> dict[str, Any]:
    """Validate and normalize current turn text and image path buffer.

    TODO(technique):
    - Normalize empty strings to `None`.
    - Enforce max upload count (`MAX_IMAGES_PER_TURN`).
    - Validate image extensions and file existence for `pending_image_paths`.
    """
    text_normalized = state.get("pending_user_text", "")
    if text_normalized:
        if text_normalized.strip() == "":
            text_normalized = None

    image_paths = state.get("pending_image_paths") or []
    paths_resolved: list[str] = []
    for path in image_paths:
        path_resolved = Path(path).resolve()
        if path_resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            return {"error": f"File at {path_resolved} is not a valid image type."}
        if not path_resolved.is_file():
            return {"error": f"Image at {path_resolved} could not be found."}
        paths_resolved.append(str(path_resolved))

    if len(paths_resolved) > MAX_IMAGES_PER_TURN:
        return {"error": "More than 3 images uploaded."}
    return {
        "pending_user_text": text_normalized,
        "pending_image_paths": paths_resolved,
    }


def prepare_images_for_turn(state: AgentState) -> dict[str, Any]:
    """
    Resize + encode images for the current turn only.
    """
    imgs_b64: list[str] = []

    img_paths = state.get("pending_image_paths") or []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            return {"error": f"Image at {path} could not be read."}
        try:
            img = scale_to_max_dimension(img, MAX_IMAGE_EDGE_PX)
        except Exception as e:
            return {"error": f"Image at {path} could not be scaled: {e}"}

        retval, buffer = cv2.imencode(".jpg", img)
        if not retval:
            return {"error": f"Image at {path} could not be encoded."}
        img_b64 = base64.b64encode(buffer.tobytes())
        imgs_b64.append(img_b64.decode("ascii"))
    return {"pending_image_b64": imgs_b64}


def append_user_message(state: AgentState) -> dict[str, Any]:
    """Append one user message in Ollama message shape.

    TODO(technique):
    - Message shape: `role`, `content`, optional `images` (list of b64 strings).
    - Include `images` key only when there are turn uploads.
    - Keep history append-only; do not mutate prior messages.
    """
    images = state.get("pending_image_b64") or []
    text = state.get("pending_user_text") or ""

    blocks: list[msg_content.ContentBlock] = [msg_content.create_text_block(text)]
    for img in images:
        blocks.append(
            msg_content.create_image_block(base64=img, mime_type="image/jpeg")
        )
    message: HumanMessage = HumanMessage(content_blocks=blocks)
    return {"messages": message}


def call_vlm(state: AgentState, settings: OllamaSettings) -> dict[str, Any]:
    """Call LLaVA and return assistant message update.

    TODO(technique):
    - Send full `messages` history to Ollama.
    - Append assistant output to `messages` and mirror into `last_model_message`.
    """
    outMessage: AIMessage = chat_with_llava(state.get("messages") or [], settings)
    return {"messages": outMessage, "last_model_message": outMessage}


def route_after_ingest(state: AgentState) -> str:
    """Route based on whether this turn includes image uploads."""
    if state.get("error"):
        return "error"
    if len(state.get("pending_image_paths", [])) == 0:
        return "append_user_message"
    return "prepare_images_for_turn"


def handle_error(state: AgentState) -> dict[str, Any]:
    print(f"Error: {state.get('error')}")
    return {"last_model_message": None}


def route_on_error(state: AgentState) -> str:
    return "error" if state.get("error") else "ok"


def clear_turn_buffers(state: AgentState) -> dict[str, Any]:
    """Clear transient upload/text buffers after one round trip.

    TODO(technique):
    - Reset `pending_user_text`, `pending_image_paths`, `pending_image_b64`.
    - Keep `messages` intact.
    """
    return {
        "pending_user_text": None,
        "pending_image_paths": [],
        "pending_image_b64": [],
        "error": None,
    }
