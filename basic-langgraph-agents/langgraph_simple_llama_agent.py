# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to the LLM llama-3.2-1B-Instruct, then prints the
# what the LLM returns as text to stdout.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
# The program gets the LLM llama-3.2-1B-Instruct from Hugging Face and wraps
# it for LangChain using HuggingFacePipeline.
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
import re
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]


# Determine the best available device for inference


# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.


class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
        - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - messages: Chat history (updated via add_messages reducer)

    State Flow:
    1. Initial state: all fields empty/default
    2. After get_user_input: user_input and should_exit are populated
    3. After call_llm: assistant message(s) appended to messages
    4. After print_response: state unchanged (node only reads, doesn't write)

    The graph loops continuously:
        get_user_input -> [conditional] -> call_llm -> print_response -> get_user_input
                              |
                              +-> END (if user wants to quit)
    """

    should_exit: bool
    target_models: list[str]
    is_verbose: bool
    parallel: bool
    shared: bool
    messages: Annotated[list[AnyMessage], add_messages]


def create_llm(model_id="meta-llama/Llama-3.2-1B-Instruct", name: str = "llama"):
    """
    Create and configure the LLM using HuggingFace's transformers library.
    Downloads llama-3.2-1B-Instruct from HuggingFace Hub by default and wraps it
    for use with LangChain via HuggingFacePipeline.


    """
    # Get the optimal device for inference
    device = get_device()

    print(f"Loading model: {model_id}")
    print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,  # Enable sampling for varied responses
        temperature=0.7,  # Controls randomness (lower = more deterministic)
        top_p=0.95,  # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
        return_full_text=False,
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), name=name)

    print("Model loaded successfully!")
    return llm


def create_graph(llms, checkpointer: BaseCheckpointSaver | None = None):
    """
    Create the LangGraph state graph with three separate nodes:
    1. get_user_input: Reads input from stdin
    2. call_llm: Sends input to the LLM and gets response
    3. print_response: Prints the LLM's response to stdout

    Graph structure with conditional routing and internal loop:
        START -> get_user_input -> [conditional] -> call_llm -> print_response -+
                       ^                 |                                       |
                       |                 +-> END (if user wants to quit)         |
                       |                                                         |
                       +---------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    # This node reads a line of text from stdin and updates the state.
    # State changes:
    #   - user_input: Set to the text entered by the user
    #   - should_exit: Set to True if user typed quit/exit/q, False otherwise
    #   - messages: Unchanged (not modified by this node)

    def strip_wake_word(text: str, names: list[str]) -> str:
        if not names:
            return text
        names_pattern = "|".join(re.escape(name) for name in names)
        pattern = rf"^\s*hey\s*(?:{names_pattern})\b[!.,]?\s*"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    def get_user_input(state: AgentState) -> dict:
        """
        Node that prompts the user for input via stdin.

        Reads state: Nothing (fresh input each iteration)
        Updates state:
            - user_input: The raw text entered by the user
            - should_exit: True if user wants to quit, False otherwise
        """
        # Display banner before each prompt

        # assume no target models
        target_models = []

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            return {
                "should_exit": True,  # Signal to exit the graph
                "target_models": target_models,
            }

        # Verbose ON
        if user_input.lower() in ["verbose", "v"]:
            print("Verbose output on")
            return {
                "is_verbose": True,
                "target_models": target_models,
            }

        # Verbose OFF
        if user_input.lower() in ["quiet"]:
            print("Verbose output off")
            return {
                "user_input": user_input,
                "should_exit": False,
                "is_verbose": False,
                "target_models": target_models,
            }

        # Parallel mode TOGGLE
        if user_input.lower() in ["parallel", "p"]:
            if not state.get("shared", True):
                print("Parallel is disabled while shared history is OFF")
                return {
                    "parallel": False,
                    "target_models": target_models,
                }
            print(
                f"Parallel requests toggled {'OFF' if state.get('parallel') else 'ON'}"
            )
            return {
                "parallel": not state.get("parallel"),
                "target_models": target_models,
            }

        # Shared history TOGGLE
        if user_input.lower() in ["shared", "s"]:
            shared = state.get("shared", True)
            new_shared = not shared
            print(f"Shared history toggled {'OFF' if shared else 'ON'}")
            if not new_shared and state.get("parallel"):
                print("Shared history OFF forces parallel OFF")
                return {
                    "shared": new_shared,
                    "parallel": False,
                    "target_models": target_models,
                }
            return {
                "shared": new_shared,
                "target_models": target_models,
            }

        if state.get("parallel"):
            target_models = ["qwen", "llama"]
        # Check if no inputs to target
        elif "hey qwen" in user_input:
            target_models = ["qwen"]
        elif "hey llama" in user_input:
            target_models = ["llama"]

        # Valid input - continue to LLM
        print("proceeding to llm")
        return {
            "should_exit": False,  # Signal to proceed to LLM
            "target_models": target_models,
            "messages": [
                HumanMessage(
                    content=f"(user) {strip_wake_word(user_input, ['qwen', 'llama'])}",
                    name="user",
                    additional_kwargs={"targets": target_models},
                )
            ],
        }

    # =========================================================================
    # NODE 2: call_llm
    # =========================================================================
    # This node takes the user input from state, sends it to the LLM,
    # and stores the response back in state.
    # State changes:
    #   - user_input: Unchanged (read only)
    #   - should_continue: Unchanged (read only)
    #   - messages: Appends the LLM's generated response(s)
    def call_llm(state: AgentState) -> dict:
        """
        Node that invokes the LLM with the user's input.

        Reads state:
            - user_input: The text to send to the LLM
        Updates state:
            - messages: The LLM-generated response(s)
        """

        def strip_prefixes(text: str) -> str:
            prefixes = ("(user) ", "(llama) ", "(qwen) ")
            stripped = text
            while True:
                for prefix in prefixes:
                    if stripped.startswith(prefix):
                        stripped = stripped[len(prefix) :]
                        break
                else:
                    return stripped

        def normalize_message(message: AnyMessage) -> AnyMessage:
            if isinstance(message, HumanMessage):
                if not isinstance(message.content, str):
                    return message
                return HumanMessage(
                    content=strip_prefixes(message.content),
                    name=message.name,
                    additional_kwargs=message.additional_kwargs,
                )
            if isinstance(message, AIMessage):
                if not isinstance(message.content, str):
                    return message
                return AIMessage(
                    content=strip_prefixes(message.content),
                    name=message.name,
                    additional_kwargs=message.additional_kwargs,
                    response_metadata=message.response_metadata,
                    tool_calls=message.tool_calls,
                    invalid_tool_calls=message.invalid_tool_calls,
                )
            return message

        # Invoke the LLM and get the response
        messages = state.get("messages", None)
        if messages is None:
            raise Exception(
                "call_llm with empty messages is not supposed to be possible"
            )

        response = []
        for model in state.get("target_models"):
            llm = llms.get(model, None)
            if llm is None:
                raise Exception(f"Invalid model invoked: {model}")
            llm_chf: ChatHuggingFace = llm.get("chf", None)
            if llm_chf is None:
                raise Exception(f'No ChatHuggingFace object found at: {model}["chf"]')
            llm_sysprompt = llm.get("sysprompt", None)
            if llm_sysprompt is None:
                raise Exception(f"No system prompt found for model: {model}")
            if state.get("shared", True):
                model_messages = messages
            else:
                model_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        targets = msg.additional_kwargs.get("targets")
                        if targets is None:
                            continue
                        if isinstance(targets, str):
                            if targets == model:
                                model_messages.append(msg)
                            continue
                        if isinstance(targets, list):
                            if len(targets) == 1 and targets[0] == model:
                                model_messages.append(msg)
                    elif isinstance(msg, AIMessage) and msg.name == model:
                        model_messages.append(msg)
            if not model_messages or not isinstance(model_messages[-1], HumanMessage):
                if state.get("is_verbose"):
                    print(
                        f"[verbose] skipping {model}: no user message after filtering"
                    )
                continue
            input: list[AnyMessage] = [
                llm_sysprompt,
                *[normalize_message(m) for m in model_messages],
            ]
            if state.get("is_verbose"):
                print(f"[verbose] {[str(x.content)[:8] + '...' for x in input]}")

            raw = llm_chf.invoke(input)
            content = raw.content
            if isinstance(content, str):
                content = strip_prefixes(content)
            tagged = AIMessage(content=f"({model}) {content}", name=model)

            response.append(tagged)

        if isinstance(response, list):
            return {"messages": response}
        raise Exception("how would this even be possible?")

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    # This node reads the LLM response from state and prints it to stdout.
    # State changes:
    #   - No changes (this node only reads state, doesn't modify it)
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the LLM's response to stdout.

        Reads state:
            - messages: The messages to print
        Updates state:
            - Nothing (returns empty dict, state unchanged)
        """

        recent_ai_messages = (
            msg
            for msg in reversed(state.get("messages", []))
            if isinstance(msg, AIMessage)
        )
        last_two = (next(recent_ai_messages, None), next(recent_ai_messages, None))

        print("\n" + "-" * 50)
        print("LLM Response:")
        print("-" * 50)
        for model in state.get("target_models"):
            found = False
            for msg in last_two:
                if msg is not None:
                    if msg.name == model:
                        found = True
                        print(msg.content)
                        break
            if not found:
                print(f"({model}) (no response)")

        if state.get("is_verbose"):
            print("[verbose] routing print_response -> get_user_input\n\n")

        # Return empty dict - no state updates from this node
        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    # This function examines the state and determines which node to go to next.
    # It's used for conditional edges after get_user_input.
    # Two possible routes:
    #   1. User wants to quit -> END
    #   2. User entered any input -> proceed to call_llm
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.

        Examines state:
            - should_exit: If True, terminate the graph

        Returns:
            - "__end__": If user wants to quit
            - "call_llm": If user provided any input (including empty)
        """

        # Check if user wants to exit
        if state.get("should_exit", False):
            if state.get("is_verbose"):
                print("[verbose] routing get_user_input -> END\n\n")
            return END

        # Checks if a valid input went through
        last_message = (state.get("messages") or [None])[-1]
        user_input = last_message if isinstance(last_message, HumanMessage) else ""

        if user_input in [
            "",
            "verbose",
            "quiet",
            "p",
            "v",
            "parallel",
            "p",
            "shared",
            "s",
        ]:
            if state.get("is_verbose"):
                print("[verbose] routing get_user_input -> get_user_input\n\n")
            return "get_user_input"

        if len(state.get("target_models")) == 0:
            return "get_user_input"

        # Default: Proceed to LLM
        if state.get("is_verbose"):
            print("[verbose] routing get_user_input -> call_llm\n\n")
        return "call_llm"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    # Create a StateGraph with our defined state structure
    graph_builder = StateGraph(AgentState)

    # Add all three nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    # Define edges:
    # 1. START -> get_user_input (always start by getting user input)
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> call_llm OR END
    #    Uses route_after_input to decide based on state.should_exit
    graph_builder.add_conditional_edges(
        "get_user_input",  # Source node
        route_after_input,  # Routing function that examines state
        {
            "get_user_input": "get_user_input",
            "call_llm": "call_llm",  # Any input -> proceed to LLM
            END: END,  # Quit command -> terminate graph
        },
    )

    # 3. call_llm -> print_response (always print after LLM responds)
    graph_builder.add_edge("call_llm", "print_response")

    # 4. print_response -> get_user_input (loop back for next input)
    #    This creates the continuous loop - after printing, go back to get more input
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


def main():
    """
    Main function that orchestrates the simple agent workflow:
        1. Initialize the LLM
    2. Create the LangGraph
    3. Save the graph visualization
    4. Run the graph once (it loops internally until user quits)

    The graph handles all looping internally through its edge structure:
        - get_user_input: Prompts and reads from stdin
    - call_llm: Processes input through the LLM
    - print_response: Outputs the response, then loops back to get_user_input

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default="default")
    args = parser.parse_args()

    print("=" * 50)
    print("LangGraph Simple Agent with Llama-3.2-1B-Instruct")
    print("=" * 50)
    print()

    # Step 1: Create and configure the LLMs
    llm_llama = create_llm(name="llama")
    llm_qwen = create_llm(model_id="Qwen/Qwen2.5-0.5B-Instruct", name="qwen")

    # Wrap pipelines

    # Step 2: Build the LangGraph with the LLM

    print("\nCreating LangGraph...")
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        graph = create_graph(
            {
                "llama": {
                    "chf": llm_llama,
                    "sysprompt": SystemMessage("""
                       You are Llama-3, a helpful AI chat agent.
                       Messages are automatically prefixed like (user), (llama), or (qwen).
                       Do not include any prefix in your replies.
                       Speak only as yourself (llama) and do not impersonate the user or qwen.
                       """),
                },
                "qwen": {
                    "chf": llm_qwen,
                    "sysprompt": SystemMessage("""
                       You are Qwen2, a helpful AI chat agent.
                       Messages are automatically prefixed like (user), (llama), or (qwen).
                       Do not include any prefix in your replies.
                       Speak only as yourself (qwen) and do not impersonate the user or llama.
                       """),
                },
            },
            checkpointer=checkpointer,
        )
        print("Graph created successfully!")

        # Step 3: Save a visual representation of the graph before execution
        # This happens BEFORE any graph execution, showing the graph structure
        print("\nSaving graph visualization...")
        save_graph_image(graph)

        # Step 4: Run the graph - it will loop internally until user quits
        # Create initial state with empty/default values
        # The graph will loop continuously, updating state as it goes:
        #   - get_user_input displays banner, populates user_input and should_exit
        #   - call_llm appends AI messages
        #   - print_response displays output, then loops back to get_user_input
        initial_state: AgentState = {
            "should_exit": False,
            "target_models": [],
            "is_verbose": False,
            "parallel": False,
            "shared": True,
            "messages": [],
        }

        # Single invocation - the graph loops internally via print_response -> get_user_input
        # The graph only exits when route_after_input returns END (user typed quit/exit/q)
        config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}
        graph.invoke(initial_state, config=config)


# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()
