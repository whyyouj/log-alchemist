import pandas as pd
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai
# DF = pd.read_csv("../EDA/data/mac/Mac_2k.log_structured.csv")

def start_agent(state: list):
    print('start agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3.1', df=df)
    out = llm.prompt_agent(query=query)
    return {"agent_out": out}

def router(state: list):
    print("routing")
    llm = Agent_Ai(model = 'llama3.1')
    action = state["agent_out"]
    out = llm.query_agent(query = action + "\n Is the code related to the question. Answer with a yes or a no only.")
    if 'yes' in out.lower():
        # return 'python_agent'
        return "python_pandas_ai"
    else:
        return 'final_agent'
    
def python_pandas_ai(state:list):
    print('pandas ai agent')
    llm = state['pandas']
    query = state['input']
    out = llm.chat(query)
    return {"agent_out": out}

def final_agent(state:list):
    print("final agent")
    llm = Agent_Ai(model = "llama3.1")
    query = state['input']
    out = llm.query_agent(query=query)
    return {"agent_out":out}
    