"""LangGraph nodes for Exercise 1.

Best-practice startup notes from LangGraph docs:
- Keep nodes single-purpose: validate input, transform state, or call model.
- Return partial state updates from each node.
- Keep message history append-only unless running an explicit summary/trim strategy.


"""

from __future__ import annotations

from typing import Any

from cv2.typing import MatLike
from langchain_core.messages import HumanMessage

from .ollama_client import OllamaSettings
from .state import AgentState
from pathlib import Path
import cv2
import base64
from util.imgUtils import scale_to_max_dimension

MAX_IMAGES_PER_TURN = 3
MAX_IMAGE_EDGE_PX = 720
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


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

    image_paths = state.get("pending_image_paths")
    paths_resolved: list[Path] = []
    for path in image_paths:
        path_resolved = Path(path).resolve()
        if path_resolved.suffix not in IMAGE_EXTENSIONS:
            return {"error": f"File at {path_resolved} is not a valid image type."}
        if not path_resolved.is_file():
            return {"error": f"Image at {path_resolved} could not be found."}
        paths_resolved.append(path_resolved)

    if len(paths_resolved) > MAX_IMAGES_PER_TURN:
        return {"error": "More than 3 images uploaded."}
    return {"pending_user_text": text_normalized, "pending_image_paths": paths_resolved}


def prepare_images_for_turn(state: AgentState) -> dict[str, Any]:
    """
    Resize + encode images for the current turn only.
    """
    imgs_b64: list[bytes] = []

    img_paths = state.get("pending_image_paths")
    for path in img_paths:
        img = cv2.imread(path)
        if not img:
            return {"error": f"Image at {path} could not be read."}
        try:
            img = scale_to_max_dimension(img, MAX_IMAGE_EDGE_PX)
        except Exception as e:
            return {"error": f"Image at {path} could not be scaled: {e}"}

        retval, buffer = cv2.imencode(".jpg", img)
        if not retval:
            return {"error": f"Image at {path} could not be encoded."}
        img_b64 = base64.b64encode(buffer.tobytes())
        imgs_b64.append(img_b64)
    return {"pending_image_b64": imgs_b64}


def append_user_message(state: AgentState) -> dict[str, Any]:
    """Append one user message in Ollama message shape.

    TODO(technique):
    - Message shape: `role`, `content`, optional `images` (list of b64 strings).
    - Include `images` key only when there are turn uploads.
    - Keep history append-only; do not mutate prior messages.
    """
    images = state.get("pending_image_b64", [])
    message: HumanMessage = HumanMessage(
        state.get("pending_user_text", ""), additional_kwargs={"images": images}
    )
    return {"messages": message}


def call_vlm(
    state: AgentState, settings: OllamaSettings | None = None
) -> dict[str, Any]:
    """Call LLaVA and return assistant message update.

    TODO(technique):
    - Send full `messages` history to Ollama.
    - Append assistant output to `messages` and mirror into `last_model_text`.
    """

    raise NotImplementedError("TODO: implement model invocation node")


def route_after_ingest(state: AgentState) -> str:
    """Route based on whether this turn includes image uploads."""
    if len(state.get("pending_image_paths", [])) == 0:
        return "append_user_message"
    return "prepare_images_for_turn"


def clear_turn_buffers(state: AgentState) -> dict[str, Any]:
    """Clear transient upload/text buffers after one round trip.

    TODO(technique):
    - Reset `pending_user_text`, `pending_image_paths`, `pending_image_b64`.
    - Keep `messages` intact.
    """

    raise NotImplementedError("TODO: implement turn buffer reset node")
