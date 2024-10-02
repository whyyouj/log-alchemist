import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai

graph_stage_prefix = '[STAGE]'

# def start_agent(state: list):
#     print(graph_stage_prefix, 'Start agent')
#     df = state['df']
#     query = state['input']
#     llm = Agent_Ai(model = 'llama3.1', df=df)
#     out = llm.prompt_agent(query=query)
#     return {"agent_out": out}

def router_agent(state: list):
    print(graph_stage_prefix, 'Router agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3', df=df)
    out = llm.prompt_agent(query=query)
    print('ROUTER AGENT OUT: ', out)
    return {"agent_out": out}

def router_agent_decision(state: list):
    router_out = state['agent_out']
    if 'yes' in router_out.lower():
        return 'router_summary_agent'
    else:
        return 'final_agent'

def router_summary_agent(state: list):
    print(graph_stage_prefix, 'Router summary agent')
    llm = Agent_Ai(model='llama3.1')
    query = state['input']
    query_summary = f"""
    You are suppose to determine if the <Question> is explicitly asking for a summary. When determining whether a question is asking for a summary, focus on whether the question is requesting a high-level overview of the data (summary), or if it’s asking for a specific value, action, or detail (non-summary) Always think before answering.
    
    <Question>Is this asking for a summary: {query} 
    <Thought> ...
    <Answer> should always be a Yes or No only
    """
    out = llm.query_agent(query=query_summary)
    ans = out[out.rfind('<Answer>')+ 9:]
    print('SUMMARY AGENT OUT: ', ans)
    return {"agent_out": ans}

def router_summary_agent_decision(state: list):
    router_out = state['agent_out']
    if 'yes' in router_out.lower():
        return 'python_summary_agent'
    else:
        return 'python_pandas_ai'
    
# def router(state: list):
#     print("Routing")
#     llm = Agent_Ai(model = 'llama3.1')
#     action = state["agent_out"]
#     out = llm.query_agent(query = action + "\n Is the code related to the question. Answer with a yes or a no only.")
#     if 'yes' in out.lower():
#         return "python_pandas_ai"
#     else:
#         return 'final_agent'
    
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
    print(graph_stage_prefix, 'Summary agent')
    df = state['df']
    llm = Python_Ai(model = 'mistral', df=df)
    out = llm.get_summary()
    query = state['input']
    prompt = f"""
    The following is the query from the user:
    {query}

    The following is a list of Python dictionaries, where each dictionary has a 'path' value which is a html summary report:
    {out}

    You are to respond with a code output that answers the user query. The code must not be a function and must NOT have a return statement.

    You are to following the instructions below strictly:
    - dfs: list[pd.DataFrame] is already provided.
    - Parse through the html summary report for each Python dictionary above, print the Python dictionary/dictionaries whose html summary report answers the user query.
    """
    # - dfs: list[pd.DataFrame] is already provided.
    # - summarise_df(df: pd.DataFrame) function is already provided. You MUST use this function.
    # - For each dataframe that the user wants to summarise, use the summarise_df(df: pd.DataFrame) function provided to summarise the dataframe as such: summarise_df(df), where df 
    # is the pd.DataFrame to be summarised
    # - Print the summary results for each dataframe the user wants to summarise.
    # - Import all required dependencies in the function.
    # - Generate a random id for each dataframe to differentiate them.
    out = llm.pandas_legend().chat(prompt)
    print('PYTHON SUMMARY OUT: ', out)
    return {"agent_out": out}
    
def final_agent(state:list):
    print(graph_stage_prefix, "Final agent")
    llm = Agent_Ai(model = "llama3.1")
    query = state['input']
    out = llm.query_agent(query=query)
    return {"agent_out":out}
    