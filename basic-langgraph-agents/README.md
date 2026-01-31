# LangGraph Agent Orchestration: Llama & Qwen

This project demonstrates a multi-agent orchestration framework built with **LangGraph** and **LangChain**. It simulates a chat environment where a user can interact with two distinct LLMs--**Llama-3.2-1B** and **Qwen2.5-0.5B**--managing conversation history, routing, and state persistence.

## Architecture

The system is modeled as a StateGraph where input is processed, routed to specific models (or both in parallel), and the results are aggregated back to the user.

![LangGraph Visualization](https://drive.google.com/uc?export=view&id=1bbCh0hADw99W1ldmOnsgUXjbKd8Icn82)

*Figure 1: Generated graph structure showing the flow between input, LLM inference, and response printing.*

## Key Features

### 1. Multi-Agent Routing
The agent intelligently routes queries based on **wake words**.
- **"Hey Qwen..."** -> Routes only to the Qwen model.
- **"Hey Llama..."** -> Routes only to the Llama model.
- **Standard Input** -> Routes to the default active model(s).

### 2. Parallel Execution
A **Parallel Mode** allows the user to broadcast a single message to *both* models simultaneously to compare their responses side-by-side.

### 3. Context Management
The system handles the complexity of "multi-party" chat within standard `User`/`AI` message roles by normalizing message content. It supports two history modes:
- **Shared History (Default):** Both models see the full conversation context, including what the other model said.
- **Isolated History:** Models only see messages specifically targeted at them.

### 4. Persistence & Recovery
Uses `SqliteSaver` to checkpoint the graph state after every step. If the program crashes or is stopped, the conversation can be resumed exactly where it left off by using the same thread ID.

## Usage

### Installation
This project uses `uv` for package management, but `pip` works as well.

```bash
pip install -r requirements.txt
```

### Running the Agent
Run the main script to download models (first run only) and start the chat loop:

```bash
python langgraph_simple_agent.py
```

To resume a specific conversation thread:

```bash
python langgraph_simple_agent.py --thread-id "my-conversation-1"
```

### Runtime Commands
While in the chat loop, you can type these commands instead of a message to change the agent's behavior:

| Command | Action |
| :--- | :--- |
| `v` / `verbose` | Turn on verbose node tracing (shows internal state transitions). |
| `quiet` | Turn off verbose tracing. |
| `p` / `parallel` | Toggle **Parallel Mode** (Run both models on next input). |
| `s` / `shared` | Toggle **Shared History** (Allow models to see each other's replies). |
| `q` / `quit` | Save state and exit the program. |

## Project Structure

- `langgraph_simple_agent.py`: Main application logic, graph definition, and node functions.
- `checkpoints.sqlite`: SQLite database storing conversation history and state snapshots.
- `lg_graph.png`: Automatically generated visualization of the LangGraph structure.
