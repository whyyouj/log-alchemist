import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai
# DF = pd.read_csv("../EDA/data/mac/Mac_2k.log_structured.csv")

graph_stage_prefix = '[STAGE]'

def start_agent(state: list):
    print(graph_stage_prefix, 'start agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3.1', df=df)
    out = llm.prompt_agent(query=query)
    return {"agent_out": out}

def router_agent(state: list):
    print(graph_stage_prefix, 'router agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3.1', df=df)
    out = llm.prompt_agent(query=query)
    print('ROUTER AGENT OUT: ', out)
    return {"agent_out": out}

def router_agent_decision(state: list):
    router_out = state['agent_out']
    if 'yes' in router_out.lower():
        return 'python_pandas_ai'
    else:
        return 'final_agent'

def router(state: list):
    print("routing")
    llm = Agent_Ai(model = 'llama3.1')
    action = state["agent_out"]
    out = llm.query_agent(query = action + "\n Is the code related to the question. Answer with a yes or a no only.")
    if 'yes' in out.lower():
        return "python_pandas_ai"
    else:
        return 'final_agent'
    
def python_pandas_ai(state:list):
    print(graph_stage_prefix, 'pandas ai agent')
    llm = state['pandas']
    query = state['input']
    out = llm.chat(query)
    return {"agent_out": out}

def final_agent(state:list):
    print(graph_stage_prefix, "final agent")
    llm = Agent_Ai(model = "llama3.1")
    query = state['input']
    out = llm.query_agent(query=query)
    return {"agent_out":out}
    