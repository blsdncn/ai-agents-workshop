"""
Gradio Quick Start Template
"""

from uuid import uuid4
import gradio as gr
import os

from langchain_core.messages import AIMessage, convert_to_openai_messages
from langgraph.graph.state import CompiledStateGraph, RunnableConfig
from agent.state import AgentState, initial_state
from agent.graph import build_graph

# ============================================================
# Your processing functions go here
# ============================================================


def process_single_file(file):
    """Handle a single uploaded file. `file` is a filepath string."""
    if file is None:
        return "No file selected."

    size = os.path.getsize(file)
    name = os.path.basename(file)
    return f"File: {name}\nPath: {file}\nSize: {size:,} bytes"


def process_multiple_files(files):
    """Handle multiple uploaded files. `files` is a list of filepath strings."""
    if not files:
        return "No files selected."

    lines = [f"Selected {len(files)} file(s):\n"]
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f)
        lines.append(f"  • {name} ({size:,} bytes)")
    return "\n".join(lines)


# ============================================================
# Build the Gradio UI
# ============================================================


def create_app():
    flow: CompiledStateGraph = build_graph()
    with gr.Blocks(title="Gradio Quick Start") as app:
        # gr.Markdown("# Gradio Quick Start Demo")
        session_state = gr.State({"session_id": str(uuid4()), "initialized": False})

        def one_turn(message, _, session_state):
            updated_session_state = dict(session_state)
            if isinstance(message, str):
                print("Retry: image data might have been lost.")
                user_text = message
                user_files = []
            else:
                user_text = message.get("text") or ""
                user_files = message.get("files") or []
            if not updated_session_state.get("initialized"):
                input = {
                    **initial_state(),
                    "pending_user_text": user_text,
                    "pending_image_paths": user_files,
                }
                updated_session_state["initialized"] = True
            else:
                input = {
                    "pending_user_text": user_text,
                    "pending_image_paths": user_files,
                }
            config: RunnableConfig = {
                "configurable": {"thread_id": updated_session_state.get("session_id")}
            }
            message = flow.invoke(input, config=config).get(
                "last_model_message"
            ) or AIMessage(content="Previous request didn't return output.")
            return convert_to_openai_messages([message])[0], updated_session_state

        gr.ChatInterface(
            fn=one_turn,
            textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"]),
            additional_inputs=[session_state],
            additional_outputs=[session_state],
        )

    return app


# ============================================================
# Launch helpers
# ============================================================


def launch_app(**kwargs):
    """
    Launch the Gradio app. Pass any gr.Blocks.launch() kwargs.

    Examples:
        launch_app()                        # default (opens browser)
        launch_app(share=True)              # public URL (needed for Colab)
        launch_app(server_port=7861)        # custom port
        launch_app(inbrowser=False)         # don't auto-open browser
    """
    app = create_app()
    app.launch(**kwargs)
    return app


def is_colab():
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


# ============================================================
# Command-line entry point
# ============================================================

if __name__ == "__main__":
    if is_colab():
        launch_app(share=True)
    else:
        launch_app(inbrowser=True)
