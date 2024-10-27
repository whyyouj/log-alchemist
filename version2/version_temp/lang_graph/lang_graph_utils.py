import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai, anomaly_skill
from langchain_core.prompts import PromptTemplate
import re 
import ast

graph_stage_prefix = '[STAGE]'

def multiple_question_agent(state: list):
    print(graph_stage_prefix, "Multiple Question Parser")

    llm = Agent_Ai(model = 'jiayuan1/router_llm', df = state['df'], temperature = 0)
    out = llm.query_agent(state['input'])
    pattern = f'\[[^\]]+\]'
    try:
        parse_qns = re.findall(pattern, out)[0]
        parse_qns = parse_qns.replace("“", """\"""")
        parse_qns = parse_qns.replace("”", """\"""")
        parse_qns_list = eval(f"""{parse_qns}""")
    except:
        parse_qns_list =[state['input']]
        
    print(parse_qns_list)
    return {"input": str(parse_qns_list[0]), "remaining_qns": parse_qns_list[1:]}

def router_agent(state: list):
    print(graph_stage_prefix, 'Router Agent')
    # df = state['df']
    # query = state['input']
    # llm = Agent_Ai(model = 'llama3.1', df=df)
    # out = llm.prompt_agent(query=query)
    query = state['input']
    try:
        parse_dict = eval(query)
        qns_type = ''
        qns = ''
        
        for i in parse_dict.keys():
            qns_type = i
            qns = parse_dict[qns_type]
        if "pandas" in qns_type.lower():
            out = 'yes'
        else:
            out = 'no'
        input = qns
    except:
        out = 'yes'
        input = query
        
    print('ROUTER AGENT OUTPUT: ', out, 'TYPE', qns_type,'INPUT:', input)
    return {"agent_out": out, "input": input}

def router_agent_decision(state: list):
    # router_out = state['agent_out']
    # router_out = router_out.lower()
    # out = router_out[router_out.rfind("answer") + 5:]
    out = state['agent_out']
    if 'yes' in out.lower():
        return 'python_pandas_ai' #'router_summary_agent'
        
    else:
        return 'final_agent'

def router_summary_agent(state: list):
    print(graph_stage_prefix, 'Router summary agent')
    llm = Agent_Ai(model='jiayuan1/summary_anomaly_llm_v3')
    query = state['input']
    # query_summary = f"""
    # You are suppose to determine if the <Question> is explicitly asking for a summary. When determining whether a question is asking for a summary, focus on whether the question is requesting a high-level overview of the data (summary), or if it’s asking for a specific value, action, or detail (non-summary). Always think before answering.
    
    # <Question> Is this asking for a summary: {query} 
    # <Thought> ...
    # <Answer> Always a Yes or No only
    # """
    out = llm.query_agent(query=query)
    out = out.lower()
    # print('ROUTER SUMMARY AGENT OUTPUT: ', out)
    return {"agent_out": out}

def router_summary_agent_decision(state: list):
    router_out = state['agent_out']
    if 'summary' in router_out.lower():
        print('[INFO] Routed to Summary agent')
        return 'python_summary_agent'
    elif 'anomaly' in router_out.lower():
        print('[INFO] Routed to Anomaly agent')
        return 'python_anomaly_agent'
    else:
        print('[INFO] Routed to General agent')
        return 'python_pandas_ai'
    
def python_pandas_ai(state:list):
    print(graph_stage_prefix, 'Pandas AI agent')
    llm = state['pandas']
    query = state['input']
    # prompt = f"""
    # The following is the query from the user:
    # {query}

    # You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

    # You are to following the instructions below strictly:
    # - Any query related to Date or Time, refer to the 'Datetime' column.
    # - Any query related to ERROR, WARNING or EVENT, refer to the EventTemplate column.
    # """
    # out = llm.chat(prompt)
    out = llm.chat(query)
    return {"agent_out": out}

def python_summary_agent(state: list):
    print(graph_stage_prefix, 'Summary Agent')
    df = state['df']
    query = state['input']
    llm = Python_Ai(model = "llama3.1", df = df)
    pandasai_llm  = llm.pandas_legend_with_skill()
    prompt = f"""
    The following is the query from the user:
    {query}

    If the query contains "summary", you must only execute the code for Sweetviz and output that result only.
    If the query does not contain "summary", you are to try your best to respond to the user query with an executable code.
    """
    out = pandasai_llm.chat(prompt) #state['pandas'].chat(prompt)
    # out = summary_skill(df[1])
    print('PYTHON SUMMARY OUT: ', out)
    return {"agent_out": out}

def python_anomaly_agent(state: list):
    print(graph_stage_prefix, 'Anomaly Agent')
    df = state['df']
    # query = state['input']
    # llm = Python_Ai(model = "llama3.1", df = df)
    # pandasai_llm  = llm.pandas_legend_with_anomaly_skill() 
    # prompt = f"""
    # The following is the query from the user:
    # {query}

    # If the query contains "anomaly", you must only execute the code for anomaly and output that result only.
    # If the query does not contain "anomaly", you are to try your best to respond to the user query with an executable code.
    # """
    #prompt = 'Use your anomaly skill on the dataframe'
    #out = pandasai_llm.chat(query) 

    # Need to handle sorting of dataframes
    out = anomaly_skill(df[0])
    print('PYTHON ANOMALY OUT: ', out)
    return {"agent_out": out}

def router_python_output(state:list):
    router_out = state["agent_out"]
    if "Unfortunately, I was not able to answer your question, because of the following error:" in str(router_out):
        return "final_agent"
    else:
        return "multiple_question_parser"
    
def final_agent(state:list):
    print(graph_stage_prefix, "Final Agent")
    llm = Agent_Ai(model = "jiayuan1/nous_llm")
    query = state['input']
    previous_ans = state['all_answer']
    previous_ans_format = "Here is you knowledge base:\n"
    if previous_ans:
        for i in previous_ans:
            qns = i['qns']
            ans = i['ans']
            if isinstance(ans, pd.DataFrame):
                ans = ans.head(10)
                ans= ans.to_json()
            previous_ans_format += qns + '\n'
            previous_ans_format += f"Answer: {ans}" + "\n\n"
        previous_ans_format += "You should use the information above to answer the following question directly and concisely. If the user's question is not related to the knowledge base, answer it directly without using the knowledge base."
    else:
        previous_ans_format = ""
        
    prompt = f"""
    {previous_ans_format}
    
    Question from the user:
    {query}

    """
    out = llm.query_agent(query=prompt)
    # Try your best to answer the query. Take your time. If the query relates to any dataframe, assist accordingly to answer the query.
    # """
    # out = llm.query_agent(query=prompt)

    out = llm.query_agent(query)
    return {"agent_out":out}

def multiple_question_parser(state:list):
    print(graph_stage_prefix, "Multiple Question Parser")
    qns = state['input']
    out = state['agent_out']
    all_answer = state["all_answer"]
    qns_ans_dict = {}

    qns_ans_dict['qns'] = f'{len(all_answer)+1}. {qns}'
    qns_ans_dict['ans'] = out
    
        
    all_answer.append(qns_ans_dict)
    remaining_qns = state['remaining_qns']
    if remaining_qns:
        return {"input":str(remaining_qns[0]), "remaining_qns":remaining_qns[1:], "all_answer":all_answer}
    
    else:
        return {"input": "", "remaining_qns":[], "all_answer":all_answer}
    
def router_multiple_question(state:list):
    print(graph_stage_prefix, "Multiple Question Router")
    if state["input"]:
        return "router_agent"
    else:
        return "__end__"
    
    