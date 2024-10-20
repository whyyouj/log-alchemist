import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai

graph_stage_prefix = '[STAGE]'

def router_agent(state: list):
    print(graph_stage_prefix, 'Router Agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3.1', df=df)
    out = llm.prompt_agent(query=query)
    print('ROUTER AGENT OUTPUT: ', out)
    return {"agent_out": out}

def router_agent_decision(state: list):
    router_out = state['agent_out']
    router_out = router_out.lower()
    out = router_out[router_out.rfind("answer") + 5:]
    if 'yes' in out.lower():
        return 'router_summary_agent'
    else:
        return 'final_agent'

def router_summary_agent(state: list):
    print(graph_stage_prefix, 'Router summary agent')
    llm = Agent_Ai(model='llama3.1')
    query = state['input']
    query_summary = f"""
    You are suppose to determine if the <Question> is explicitly asking for a summary. When determining whether a question is asking for a summary, focus on whether the question is requesting a high-level overview of the data (summary), or if it’s asking for a specific value, action, or detail (non-summary). Always think before answering.
    
    <Question> Is this asking for a summary: {query} 
    <Thought> ...
    <Answer> Always a Yes or No only
    """
    out = llm.query_agent(query=query_summary)
    out = out.lower()
    ans = out[out.rfind('answer')+ 5:]
    print('ROUTER SUMMARY AGENT OUTPUT: ', out)
    return {"agent_out": out}

def router_summary_agent_decision(state: list):
    router_out = state['agent_out']
    if 'yes' in router_out.lower():
        return 'python_summary_agent'
    else:
        return 'python_pandas_ai'
    
def python_pandas_ai(state:list):
    print(graph_stage_prefix, 'Pandas AI agent')
    llm = state['pandas']
    query = state['input']
    prompt = f"""
    The following is the query from the user:
    {query}

    You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

    You are to following the instructions below strictly:
    - dfs: list[pd.DataFrame] is already provided.
    - Any query related to Date or Time, refer to the 'Datetime' column.
    - Any query related to ERROR, WARNING or EVENT, refer to the EventTemplate column.
    """
    out = llm.chat(prompt)
    return {"agent_out": out}

def python_summary_agent(state: list):
    print(graph_stage_prefix, 'Summary Agent')
    df = state['df']
    query = state['input']
    llm = Python_Ai(model = "llama3.1", df = df)
    pandasai_llm  = llm.pandas_legend_with_summary_skill()
    prompt = f"""
    The following is the query from the user:
    {query}

    If the query contains "summary", you must only execute the code for Sweetviz and output that result only.
    If the query does not contain "summary", you are to try your best to respond to the user query with an executable code.
    """
    out = pandasai_llm.chat(prompt) #state['pandas'].chat(prompt)
    print('PYTHON SUMMARY OUT: ', out)
    return {"agent_out": out}

def router_python_output(state:list):
    router_out = state["agent_out"]
    if "Unfortunately, I was not able to answer your question, because of the following error:" in str(router_out):
        return "final_agent"
    else:
        return "__end__"
    
def final_agent(state:list):
    print(graph_stage_prefix, "Final Agent")
    llm = Agent_Ai(model = "llama3.1")
    query = state['input']
    out = llm.query_agent(query=query)
    return {"agent_out":out}
    