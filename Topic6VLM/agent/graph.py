"""Graph assembly for Exercise 1.

Reference practice from LangGraph docs:
- Define explicit entry point.
- Keep conditional edges focused and testable.
- Compile once and reuse graph object.
"""

from __future__ import annotations

from typing import Any

from .nodes import (
    append_user_message,
    call_vlm,
    clear_turn_buffers,
    ingest_user_turn,
    prepare_images_for_turn,
    route_after_ingest,
)


def build_graph() -> Any:
    """Build and compile the LangGraph workflow.

    TODO(packages):
    - `langgraph.graph.StateGraph`
    - constants: `START`, `END`

    TODO(technique):
    - Flow: START -> ingest_user_turn -> conditional(router)
      - prepare_images_for_turn -> append_user_message -> call_vlm -> clear_turn_buffers -> END
      - append_user_message -> call_vlm -> clear_turn_buffers -> END
    - Add small unit tests for routing branches before model integration.
    """

    _ = (
        ingest_user_turn,
        prepare_images_for_turn,
        append_user_message,
        route_after_ingest,
        call_vlm,
        clear_turn_buffers,
    )
    raise NotImplementedError("TODO: implement graph compile step")
