from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator
from langgraph.graph import StateGraph, END, START
import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python_agent.python_ai import Python_Ai
from regular_agent.agent_ai import Agent_Ai
from lang_graph.lang_graph_utils import python_pandas_ai, final_agent, router_agent, router_agent_decision, router_summary_agent, router_summary_agent_decision, router_python_output, python_summary_agent, multiple_question_agent, multiple_question_parser, router_multiple_question

class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    pandas: Python_Ai
    df: pd.DataFrame
    remaining_qns: list
    all_answer: list
    
class Graph:
    def __init__(self, pandas_llm, df):
        self.pandas = pandas_llm
        self.df = df
        self.qns=''
        self.graph = Graph.get_graph()
    
    @staticmethod
    def get_graph():
        graph = StateGraph(AgentState)

        # LangGraph Nodes
        graph.add_node("multiple_question_agent", multiple_question_agent) # Breakdown if question is about multiple questions
        graph.add_node('router_agent', router_agent) # Determining if question is related to dataset
        graph.add_node('router_summary_agent', router_summary_agent) # Determining if question is asking for summary
        graph.add_node("python_pandas_ai", python_pandas_ai) # Answering specific dataset related questions
        graph.add_node("python_summary_agent", python_summary_agent) # Answering summary related questions by outputting sweetviz framework
        graph.add_node('final_agent', final_agent) 
        graph.add_node("multiple_question_parser", multiple_question_parser)

        # LangGraph Edges
        graph.add_edge(START, 'multiple_question_agent')
        graph.add_edge("multiple_question_agent", "router_agent") # Initialising LangGraph
        graph.add_conditional_edges( 
            "router_agent",
            router_agent_decision,
            {
                "router_summary_agent":"router_summary_agent",
                "final_agent":"final_agent"
            }
        )
        graph.add_conditional_edges(
            "router_summary_agent",
            router_summary_agent_decision,
            {
                "python_summary_agent":"python_summary_agent",
                "python_pandas_ai":"python_pandas_ai"
            }
        )
        graph.add_conditional_edges(
            "python_pandas_ai",
            router_python_output,
            {
                "final_agent":"final_agent",
                "multiple_question_parser":"multiple_question_parser"
            }
        )      
        graph.add_conditional_edges(
            "python_summary_agent",
            router_python_output,
            {
                "final_agent":"final_agent",
                "multiple_question_parser":"multiple_question_parser"
            }
        )
        # graph.add_edge("python_pandas_ai", END)
        graph.add_edge("final_agent", "multiple_question_parser")
        
        graph.add_conditional_edges(
            "multiple_question_parser",
            router_multiple_question,
            {
                "router_agent":"router_agent",
                "__end__":"__end__"
            }
        )
        
        runnable = graph.compile()

        return runnable
    
    def run(self, query):
        # llm = Agent_Ai(model='mistral', temperature=0)
        # prompt = f"""The user has asked: '{query}'.
        # Determine if this question is asking to "try again", "retry", or something with a similar meaning related to repeating an action.
        # If the question is about retrying, respond with 'Yes'.
        # If the question is not about retrying, respond with 'No'.
        # Only answer 'Yes' or 'No'"""
        # ans = llm.query_agent(prompt)
        # print(f'[STAGE] Try again agent:{ans}')
        # if 'yes' in ans.lower():
        #     if self.qns != '':
        #         query = self.qns
        #     else:
        #         option = ['😀','🥳','😊','🥳','🤩','😎','😄','🤭']
        #         import numpy as np
        #         num = np.random.randint(0,len(option)-1)
        #         return f"Hi! Please ask a question {option[num]}"
        # else:
        #     self.qns = query
        if self.qns == '':
            self.qns = 'No previous question'
            
        runnable = self.graph
        out = runnable.invoke({"input":f"{query}", "pandas":self.pandas, "df":self.df, "remaining_qns":[], "all_answer":[]})
        self.qns = query
        return out['all_answer']
    
    def show(self):
        from PIL import Image as PILImage
        import io
        import os
        runnable = self.graph
        png_data = runnable.get_graph().draw_png()
        image = PILImage.open(io.BytesIO(png_data))
        os.makedirs("./image", exist_ok=True)
        image.save("./image/lang_chain_graph_pandas_new2.png")
        return "./image/lang_chain_graph_pandas_new2.png"
    
    def create_graph():
        global global_graph
        df = [pd.read_csv('../../../data/Mac_2k.log_structured.csv')]
        pandas_ai = Python_Ai(model='llama3.1',df=df).pandas_legend()
        global_graph = Graph(pandas_llm=pandas_ai, df=df)
        return global_graph
    
if __name__ == "__main__":
    graph = Graph.create_graph()
    # graph.show()
    graph.run('how many rows are there')
