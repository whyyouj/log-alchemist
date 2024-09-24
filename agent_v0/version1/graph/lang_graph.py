from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator
from langgraph.graph import StateGraph, END
from lang_graph_utils import start_agent, python_agent, python_agent_2, graph_agent, python_final_agent, final_agent, router, python_router, python_pandas_ai
    
class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    graph_out: str



graph = StateGraph(AgentState)

graph.add_node("start_agent", start_agent)
# graph.add_node("python_agent", python_agent)
# graph.add_node("python_agent_2", python_agent_2)
graph.add_node('final_agent', final_agent)
graph.add_node("python_pandas_ai", python_pandas_ai)
graph.add_node("python_final_agent", python_final_agent)
# graph.add_node("graph_agent", graph_agent)

graph.set_entry_point("start_agent")
# graph.add_conditional_edges(
#     "start_agent",
#     router,
#     {
#         "python_agent":"python_agent",
#         "final_agent":"final_agent"
#     }

# )

graph.add_conditional_edges(
    "start_agent",
    router,
    {
        "python_pandas_ai":"python_pandas_ai",
        "final_agent":"final_agent"
    }

)
 
# graph.add_conditional_edges(
#     "python_agent",
#     python_router, 
#     {
#         "python_agent_2" : "python_agent_2",
#         "graph_agent":"graph_agent",
#     }
# )

# graph.add_edge("python_agent_2", "graph_agent")
# graph.add_edge("graph_agent", "python_final_agent")
graph.add_edge("python_pandas_ai", "python_final_agent")
graph.add_edge("python_final_agent", END)
graph.add_edge("final_agent", END)

runnable = graph.compile()  


if __name__ == "__main__":
    from IPython.display import Image
    from PIL import Image as PILImage
    import io
    input = input("Ask a question:")
    out = runnable.invoke({"input": f"{input}"})
    print(out['agent_out'])

    # Assuming runnable.get_graph().draw_png() returns a PNG image in bytes
    # png_data = runnable.get_graph().draw_png()

    # Create a PIL image from the PNG bytes
    # image = PILImage.open(io.BytesIO(png_data))

    # Save the image to a file
    # image.save("../image/lang_chain_graph_pandas.png")  # You can specify your desired file name and format
