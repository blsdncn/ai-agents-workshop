"""Entry point for Topic 6 VLM assignments.

Currently wired as Exercise 1 scaffolding only.
"""

from agent.ui import build_app


def main() -> None:
    """Launch the Exercise 1 UI shell.

    TODO(tool):
    - Launch via `uv run python main.py`.
    - If in Colab, pass `share=True` when calling `.launch()`.

    TODO(technique):
    - Keep this file thin: UI construction in `agent/ui.py`, graph logic in `agent/graph.py`.
    """

    app = build_app()
    _ = app
    raise NotImplementedError("TODO: wire Gradio launch options")


if __name__ == "__main__":
    main()
