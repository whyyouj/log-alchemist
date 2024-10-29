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
from lang_graph.lang_graph_utils import multiple_question_agent, router_agent,router_agent_decision, python_pandas_ai, router_python_output, final_agent, multiple_question_parser, router_multiple_question

class AgentState(TypedDict):
    
    '''
    The variables that will be present in the state of the Lang Graph
    '''
    
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
        self.qns = ''
        self.graph = Graph.get_graph()
    
    @staticmethod
    def get_graph():
        
        '''
        The function that builds the graph
        '''
        
        graph = StateGraph(AgentState)

        # LangGraph Nodes
        graph.add_node("multiple_question_agent", multiple_question_agent) # Breakdown if question is about multiple questions
        graph.add_node('router_agent', router_agent) # Determining if question is related to dataset
        graph.add_node("python_pandas_ai", python_pandas_ai) # Answering specific dataset related questions
        graph.add_node('final_agent', final_agent) 
        graph.add_node("multiple_question_parser", multiple_question_parser)

        # LangGraph Edges
        graph.add_edge(START, 'multiple_question_agent') # Initialising LangGraph
        graph.add_edge("multiple_question_agent", "router_agent")
        graph.add_conditional_edges( 
            "router_agent",
            router_agent_decision,
            {
                "python_pandas_ai":"python_pandas_ai",
                "final_agent":"final_agent"
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
        graph.add_edge("final_agent", "multiple_question_parser")
        
        graph.add_conditional_edges(
            "multiple_question_parser",
            router_multiple_question,
            {
                "router_agent":"router_agent",
                "__end__":"__end__"
            }
        )
        
        # compile the graph
        runnable = graph.compile()

        return runnable
    
    def run(self, query):
        
        '''
        The function that will run a user query using the Lang Graph
        '''
        
        runnable = self.graph
        out = runnable.invoke({"input":f"{query}", "pandas":self.pandas, "df":self.df, "remaining_qns":[], "all_answer":[]})
        self.qns = query
        return out['all_answer']
    
    def show(self):
        
        '''
        Plot the Lang Graph to have a visualisation of a plot of how the graph will look
        '''
        
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
    df = [pd.read_csv('../../../data/Mac_2k.log_structured.csv')]
    pandas_ai = Python_Ai(model = "llama3.1", df=df).pandas_legend()
    graph = Graph(pandas_llm = pandas_ai, df = df)
    graph.show()
    #graph.run('how many rows are there')
