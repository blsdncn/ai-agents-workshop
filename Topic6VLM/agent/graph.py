"""Graph assembly for Exercise 1.

Reference practice from LangGraph docs:
- Define explicit entry point.
- Keep conditional edges focused and testable.
- Compile once and reuse graph object.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.ollama_client import OllamaSettings

from .nodes import (
    append_user_message,
    call_vlm,
    clear_turn_buffers,
    ingest_user_turn,
    prepare_images_for_turn,
    route_after_ingest,
)
from functools import partial

from .state import AgentState


def build_graph() -> CompiledStateGraph:
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

    settings = OllamaSettings()
    call_vlm_node = partial(call_vlm, settings=settings)

    builder = StateGraph(AgentState)
    builder.add_node(ingest_user_turn)
    builder.add_node(prepare_images_for_turn)
    builder.add_node(append_user_message)
    builder.add_node("call_vlm", call_vlm_node)
    builder.add_node(clear_turn_buffers)
    builder.add_edge(START, "ingest_user_turn")
    builder.add_conditional_edges(
        "ingest_user_turn",
        route_after_ingest,
        {
            "append_user_message": "append_user_message",
            "prepare_images_for_turn": "prepare_images_for_turn",
        },
    )
    builder.add_edge("prepare_images_for_turn", "append_user_message")
    builder.add_edge("append_user_message", "call_vlm")
    builder.add_edge("call_vlm", "clear_turn_buffers")
    builder.add_edge("clear_turn_buffers", END)
    return builder.compile()
