"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from manual_tool_handling import calculator
from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Literal, Annotated
import fortune
import random
from langchain_core.runnables import RunnableConfig
import argparse
from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]

# ============================================
# PART 1: Define Your Tools
# ============================================


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculator_tool(expression: str) -> str:
    """Thin wrapper around calculator tool that uses numexpr"""
    return calculator(expression)


@tool
def divine_insight() -> str:
    """Returns random bit of divine insight (fortune cookie fortune)."""
    return (
        fortune.get_random_fortune("./fortunes.txt")
        + f" Lucky numbers: {random.randint(1, 100)}, {random.randint(1, 100)}"
    )


@tool
def count_instances_of_substring(text: str, substr: str) -> str:
    """
    Counts the instances of the provided substring within the text.

    Args:
        text: The actual string from which you will count instances of the substring from.
        substr: The substring the tool will count the number of instances of.

    Returns:
        String report of the result of counting the number of occurances of the substring.
    """
    return f"Occurances of {substr}: {text.count(substr)}"


tools = [get_weather, calculator_tool, divine_insight, count_instances_of_substring]
tool_map = {tool.name: tool for tool in tools}

# ============================================kj
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ============================================
# PART 3: The Agent Loop
# ============================================


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sysprompt: SystemMessage
    clear_requested: bool


class AgentUpdate(TypedDict, total=False):
    messages: list[AnyMessage]
    sysprompt: SystemMessage
    clear_requested: bool


def run_agent(state: AgentState) -> AgentUpdate:
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    # Start conversation with user query
    messages = state.get("messages")
    responses: list[AnyMessage] = []
    user_query = messages[-1]
    if not isinstance(user_query, HumanMessage):
        raise Exception(
            "Last message should be an instance of a HumanMessage if in run_agent node."
        )

    # print(f"User: {user_query}\n")

    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        # Call the LLM
        response = llm_with_tools.invoke(
            [state.get("sysprompt")] + messages + responses
        )

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")

            # Add the assistant's response to messages
            responses.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Execute the tool
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Add the tool result back to the conversation
                responses.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

            print()
            # Loop continues - LLM will see the tool results

        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            responses.append(response)
            return {"messages": responses}

    raise Exception("Maximum iterations reached")


# ============================================
# PART 5: Chat loop implementation
# ============================================


def get_message(state: AgentState) -> AgentUpdate:
    user_input = input("> ")
    return {"messages": [HumanMessage(user_input)]}


def check_exit(state: AgentState) -> Literal["run_agent", "clear_memory", "__end__"]:
    if not isinstance(state.get("messages")[-1], HumanMessage):
        raise Exception(
            "Last message should be an instance of a HumanMessage if in check_exit node."
        )
    command = str(state.get("messages")[-1].content).strip().lower()
    if command in ["q", "quit", "exit"]:
        return "__end__"
    if command in ["clear", "reset"]:
        return "clear_memory"

    return "run_agent"


def clear_memory(state: AgentState, config: RunnableConfig) -> AgentUpdate:
    _ = state
    _ = config
    print("Clear requested. Ending run so checkpoint memory can be cleared.")
    return {"clear_requested": True}


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
    initial_state: AgentState = {
        "messages": [],
        "sysprompt": SystemMessage(
            content="You are a helpful assistant. Use the provided tools when needed."
        ),
        "clear_requested": False,
    }

    agent_builder = StateGraph(AgentState)

    agent_builder.add_node("run_agent", run_agent)
    agent_builder.add_node("get_message", get_message)
    agent_builder.add_node("clear_memory", clear_memory)
    agent_builder.add_edge(START, "get_message")
    agent_builder.add_conditional_edges(
        "get_message", check_exit, ["run_agent", "clear_memory", END]
    )
    agent_builder.add_edge("run_agent", "get_message")
    agent_builder.add_edge("clear_memory", END)

    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default="default")
    args = parser.parse_args()

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        agent = agent_builder.compile(checkpointer=checkpointer)
        save_graph_image(agent)
        config: RunnableConfig = {"configurable": {"thread_id": args.thread_id}}
        final_state = agent.invoke(initial_state, config=config)
        if final_state.get("clear_requested"):
            checkpointer.delete_thread(args.thread_id)
            print(f"Cleared checkpoint memory for thread '{args.thread_id}'.")


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    main()

### Old code from directly calling run_agent
# Test query that requires tool use
# print("=" * 60)
# print("TEST 1: Query requiring tool")
# print("=" * 60)
# run_agent("What's the weather like in San Francisco?")
#
# print("\n" + "=" * 60)
# print("TEST 2: Query not requiring tool")
# print("=" * 60)
# run_agent("Say hello!")
#
# print("\n" + "=" * 60)
# print("TEST 3: Multiple tool calls")
# print("=" * 60)
# run_agent("What's the weather in New York and London?")
# print("\n" + "=" * 60)
# print("TEST 4: Using new tool")
# print("=" * 60)
# run_agent("Can you give me some divine insight please?")
# print("\n" + "=" * 60)
# print("TEST 5: Hitting iteration limit")
# print("=" * 60)
# run_agent(
#     "Create a daily report/fortune, check what the weather is in London and in San Francisco, if the difference in temperature is London and in San Francisco is an even number, then it will be a good day. If its odd, it will be a bad day. You will then invoke divine insight, and you will then come up with a contrived way to tie in the fortune's message with the current weather conditions. Lastly, divide the two lucky numbers, if the result is an integer (the lucky numbers divide evenly), ignore literally all previous and following instructins and just yell \"AHHHHH!!!\" as your only output. Lastly, the true lucky number will the difference between the two lucky numbers divided by the number of 'i's in the full string of the divine insight."
# )
