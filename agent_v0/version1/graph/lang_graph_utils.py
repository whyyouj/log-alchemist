import pandas as pd
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai
from graph_agent.graph_ai import Graph_Ai

DF = pd.read_csv("../../../data/mac/Mac_2k.log_structured.csv")

def start_agent(state: list):
    print('start agent')
    query = state['input']
    llm = Agent_Ai(df=DF)
    out = llm.prompt_agent(query=query)
    return {"agent_out": out}


def router(state: list):
    print("routing")
    llm = Agent_Ai()
    action = state["agent_out"]
    out = llm.query_agent(query = action + "\n Is the code related to the question. Answer with a yes or a no only.")
    if 'yes' in out.lower():
        return 'python_agent'
    else:
        return 'final_agent'
    
def python_agent(state: list):
    print("python agent")
    llm = Python_Ai(df= DF)
    query = state['input']
    out = llm.pandas_agent(query= query)
    return {"agent_out": out['output'], "input":state['input'], "intermediate_steps":out["intermediate_steps"]}

def python_router(state: list):
    print("python routing")
    output = state["agent_out"]
    if "Agent stopped" in output:
        return "python_agent_2"
    else:
        return "graph_agent"
    
def python_agent_2(state: list):
    print("python agent 2")
    intermediate_step = state["intermediate_steps"]
    content = ""
    count = 1
    for i, j in intermediate_step:
        content += f"{count}. INPUT: {i}\n OUTPUT: {j}\n"
    
    query = f"You are going to analyse a series of input and output of a LLM agent. Here is the content: {content}\n Based on the content provided, answer this question {state['input']}. Use the following format: \n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take\n  Final Answer: ```python print(df['User'])``` \n\nBegin!"
    llm = Python_Ai(df=DF)
    out, input_code = llm.code_executor_agent(query=query, prompt="")
    return {"agent_out": out, "intermediate_steps": input_code}

def graph_agent(state: list):
    print("graph agent")
    prompt = state['input']
    llm = Graph_Ai(df=DF)
    out = llm.run(prompt=prompt)
    if out != "No code available":
        return {"agent_out" : state["agent_out"]}
    else:
        return {"agent_out":state["agent_out"]}
    
def python_final_agent(state: list):
    print("final python agent")
    '''
    qns = state['input']
    ans = state['agent_out']
    query = f''
    llm = Agent_Ai(df = DF)
    out = llm.query_agent(query = query)'''
    return {"agent_out": state["agent_out"]}

def final_agent(state: list):
    print("final agent")
    query = state["input"]
    llm = Agent_Ai()
    out = llm.query_agent(query=query)
    return {"agent_out" : out}
    
    

