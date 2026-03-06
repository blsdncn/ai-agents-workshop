from agent.state import initial_state
from agent.graph import build_graph


def testGraphOneTurn():
    flow = build_graph()
    init_state = initial_state()
    init_state["pending_user_text"] = "Describe the image."
    init_state["pending_image_paths"] = ["screenshot.png"]
    out = flow.invoke(init_state)
    print(out["messages"][-1].content)
