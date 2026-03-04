"""
Gradio Quick Start Template
============================
Supports: Command line (Mac/Win/Linux), Jupyter notebooks (local), Google Colab

Features:
  - Single file picker
  - Multi-file picker
  - Text output window

Usage:
  Command line:  python gradio_quickstart.py
  Jupyter/Colab: Run the cells below (or import and call launch_app())
"""

import gradio as gr
import os

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
    with gr.Blocks(title="Gradio Quick Start") as app:
        gr.Markdown("# Gradio Quick Start Demo")
        
        # --- Single File Picker ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Single File Picker")
                single_file = gr.File(
                    label="Choose one file",
                    file_count="single",
                    type="filepath"     # gives you the filepath as a string
                )
                btn_single = gr.Button("Process File")
            with gr.Column():
                out_single = gr.Textbox(
                    label="Output",
                    lines=5,
                    interactive=False   # read-only
                )
        
        btn_single.click(
            fn=process_single_file,
            inputs=single_file,
            outputs=out_single
        )
        
        # --- Multi-File Picker ---
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Multi-File Picker")
                multi_files = gr.File(
                    label="Choose multiple files",
                    file_count="multiple",
                    type="filepath"
                )
                btn_multi = gr.Button("Process Files")
            with gr.Column():
                out_multi = gr.Textbox(
                    label="Output",
                    lines=8,
                    interactive=False
                )
        
        btn_multi.click(
            fn=process_multiple_files,
            inputs=multi_files,
            outputs=out_multi
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
