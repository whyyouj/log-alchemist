from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator
from langgraph.graph import StateGraph, END
import pandas as pd
from python_agent.python_ai import Python_Ai
from regular_agent.agent_ai import Agent_Ai
from lang_graph.lang_graph_utils import start_agent, python_pandas_ai, final_agent, router    

class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    pandas: Python_Ai
    df: pd.DataFrame
    
class Graph:
    def __init__(self, pandas_llm, df):
        self.pandas = pandas_llm
        self.df = df
        self.qns=''
    
    def get_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("start_agent", start_agent)
        graph.add_node('final_agent', final_agent)
        graph.add_node("python_pandas_ai", python_pandas_ai)
        graph.set_entry_point("start_agent")
        graph.add_conditional_edges(
            "start_agent",
            router,
            {
                "python_pandas_ai":"python_pandas_ai",
                "final_agent":"final_agent"
            }

        )
        graph.add_edge("python_pandas_ai", END)
        graph.add_edge("final_agent", END)
        runnable = graph.compile()
        return runnable
    
    def run(self, query):
        llm = Agent_Ai(model='mistral', temperature=0)
        prompt = f"""The user has asked: '{query}'.
        Determine if this question is asking to "try again", "retry", or something with a similar meaning related to repeating an action.
        If the question is about retrying, respond with 'Yes'.
        If the question is not about retrying, respond with 'No'.
        Only answer 'Yes' or 'No'"""
        ans = llm.query_agent(prompt)
        print(f'Try again agent: {ans}')
        if 'yes' in ans.lower():
            if self.qns != '':
                query = self.qns
            else:
                option = ['😀','🥳','😊','🥳','🤩','😎','😄','🤭']
                import numpy as np
                num = np.random.randint(0,len(option)-1)
                return f"Hi! Please ask a question {option[num]}"
        else:
            self.qns = query
        runnable = self.get_graph()
        out = runnable.invoke({"input":f"{query}", "pandas":self.pandas, "df":self.df})
        return out['agent_out']
    
    def show(self):
        from PIL import Image as PILImage
        import io
        import os
        runnable = self.get_graph()
        png_data = runnable.get_graph().draw_png()
        image = PILImage.open(io.BytesIO(png_data))
        os.makedirs("./image", exist_ok=True)
        image.save("./image/lang_chain_graph_pandas.png")
        return "./image/lang_chain_graph_pandas.png"
    
if __name__ == "__main__":
    df = pd.read_csv('../../../EDA/data/mac/Mac_2k.log_structured.csv')
    pandas_ai = Python_Ai(df=df).pandas_legend()
    graph = Graph(pandas_llm= pandas_ai, df = df)
    graph.show()