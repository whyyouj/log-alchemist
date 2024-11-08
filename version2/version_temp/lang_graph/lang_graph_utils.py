# Import required libraries
import pandas as pd
import os, sys

# Add the parent directory to system path for module imports
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom AI agent modules
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai, overall_anomaly
from langchain_core.prompts import PromptTemplate
import re 
import ast

# Define constants
graph_stage_prefix = '[STAGE]'
FINAL_LLM = "jiayuan1/nous_llm"

def multiple_question_agent(state: list):
    """
    Breaks down multi-part questions into separate components.

    Function Description:
    Uses an LLM to parse complex questions into individual components and categorizes them
    based on whether they require pandas operations or explanation.

    Input:
    - state (dict): Contains:
        - df: pandas DataFrame with the data
        - input: User's multi-part question

    Output:
    - dict: Contains:
        - input: First parsed question
        - remaining_qns: List of remaining questions

    Note:
    - Returns original question as single item if parsing fails
    - Requires router_160 model to be available
        
    Example:
    Input: "How many rows are there and explain the dataset"
    Output: [{"Pandas": "How many rows are there"}, {"Explain": "Explain the dataset"}]
    """
    # Print the stage prefix for "Multiple Question Parser"
    print(graph_stage_prefix, "Multiple Question Parser")

    # Initialize the LLM Agent with specified model, data frame, and temperature
    llm = Agent_Ai(model='jiayuan1/router_160', df=state['df'], temperature=0)

    # Query the agent with the user's input from the state
    out = llm.query_agent(state['input'])

    # Define a pattern to find any text within square brackets
    pattern = r'\[[^\]]+\]'
    try:
        # Use regex to find the first instance of text within square brackets in the output
        parse_qns = re.findall(pattern, out)[0]

        # Replace special quotation marks with standard quotes
        parse_qns = parse_qns.replace("“", """\"""")
        parse_qns = parse_qns.replace("”", """\"""")

        # Evaluate the formatted string to convert it into a list of questions
        parse_qns_list = eval(f"""{parse_qns}""")
    except:
        # If an error occurs, set the question list to include only the input
        parse_qns_list = [state['input']]

    # Print the parsed question list
    print(parse_qns_list)

    # Return the first question as "input" and remaining questions as "remaining_qns"
    return {"input": str(parse_qns_list[0]), "remaining_qns": parse_qns_list[1:]}

def router_agent(state: list):
    """
    Routes questions to appropriate processing agents.

    Function Description:
    Analyzes questions to determine whether they require pandas operations
    or general explanation, routing them to the appropriate agent.

    Input:
    - state (dict): Contains input query and DataFrame

    Output:
    - dict: Contains:
        - agent_out: 'yes' for pandas queries, 'no' for others
        - input: Processed query string

    Note:
    - Defaults to 'no' if question type cannot be determined
    - Strips any dictionary formatting from input
    """
    
    # Print the stage prefix for "Router Agent"
    print(graph_stage_prefix, 'Router Agent')

     # Get the input query from the state
    query = state['input']
    qns_type = '' # Initialize question type
    qns = '' # Initialize question content
    try:
        # Attempt to evaluate the query as a dictionary
        parse_dict = eval(query)
        
        # Iterate through the keys of the dictionary to extract question type and content
        for i in parse_dict.keys():
            qns_type = i
            qns = parse_dict[qns_type]
            
        # Determine output based on the question type    
        if "pandas" in qns_type.lower():
            out = 'yes'
        else:
            out = 'no'
        input = qns
        
    except:
        # If parsing fails, set default values
        if "pandas" in query.lower():
            out = "yes"
        else:
            out = 'no'
        input = query
        
    # Print the final output of the router agent   
    print('ROUTER AGENT OUTPUT: ', out, 'TYPE', qns_type, 'INPUT:', input)
    
    # Return the output and the processed input
    return {"agent_out": out, "input": input}

def router_agent_decision(state: list):
    """
    Makes routing decision based on router agent output.

    Function Description:
    Examines the router agent's output to determine the appropriate processing path,
    directing queries either to pandas operations or general explanation handling.

    Input:
    - state (dict): Contains:
        - agent_out (str): Router agent's decision ('yes' or 'no')

    Output:
    - str: Name of next agent to handle query ('python_pandas_ai' or 'final_agent')

    Note:
    - Returns 'final_agent' if output cannot be parsed
    - Case-insensitive matching for 'yes' determination
    """

    out = state['agent_out']
    if 'yes' in out.lower():
        return 'python_pandas_ai' 
        
    else:
        return 'final_agent'

    
def python_pandas_ai(state:list):
    """
    Processes pandas-related queries using AI.

    Function Description:
    Executes pandas operations based on natural language queries,
    handling data analysis and manipulation requests.

    Input:
    - state (dict): Contains:
        - pandas: PandasAI instance
        - input: Query string
        - df: DataFrame to analyze

    Output:
    - dict: Contains:
        - agent_out: Generated pandas code or error message

    Note:
    - Returns error message if query cannot be processed
    - Code output excludes function definitions and return statements
    """
    
    print(graph_stage_prefix, 'Pandas AI agent')
    llm = state['pandas']
    query = state['input']
    prompt = f"""
        The following is the query from the user:
        {query}

        You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

        You are to following the instructions below strictly:
        - Any query related to Date or Time, refer to the 'Datetime' column.
        - Any query related to ERROR, WARNING or EVENT, refer to the EventTemplate column.
    """
    out = llm.chat(prompt)
    return {"agent_out": out}


def router_python_output(state:list):
    """
    Routes based on pandas AI agent execution success.

    Function Description:
    Analyzes the output from pandas AI agent to determine if the query was 
    successfully processed, routing to appropriate next step based on result.

    Input:
    - state (dict): Contains:
        - agent_out (str): Output from pandas AI agent execution

    Output:
    - str: Next processing stage ('final_agent' or 'multiple_question_parser')

    Note:
    - Routes to final_agent on error messages
    - Continues to question parser on successful execution
    """
    
    router_out = state["agent_out"]
    if "Unfortunately, I was not able to answer your question, because of the following error:" in str(router_out):
        return "final_agent"
    else:
        return "multiple_question_parser"
    
    
def final_agent(state:list):
    """
    Handles general explanation queries using LLM.

    Function Description:
    Processes non-pandas queries by maintaining context from previous
    Q&A pairs and generating natural language responses.

    Input:
    - state (dict): Contains:
        - input: Current query
        - all_answer: List of previous Q&A pairs

    Output:
    - dict: Contains:
        - agent_out: LLM generated response

    Note:
    - Returns direct answer if no relevant context found
    - Truncates DataFrame outputs to 10 rows in context
    """
    
    # Print the stage prefix for "Final Agent"
    print(graph_stage_prefix, "Final Agent")
        
    # Initialize the LLM agent with the specified model
    llm = Agent_Ai(model = FINAL_LLM)
    
    # Retrieve the user's query and all previous answers from the state
    query = state['input']
    previous_ans = state['all_answer']
    
    # Initialize a string to format previous answers as a knowledge base
    previous_ans_format = "Here is you knowledge base:\n"
    
    # If there are previous answers, format each as Q&A pairs to add to the knowledge base
    if previous_ans:
        for i in previous_ans:
            qns = i['qns']
            ans = i['ans']
            
            # If the answer is a DataFrame, convert to JSON (limited to the first 10 rows for brevity)
            if isinstance(ans, pd.DataFrame):
                try:
                    ans = ans.head(10)
                    ans= ans.to_json()
                except:
                    # If conversion fails, keep the answer as it is
                    ans = ans
            
            # Add question and answer to the knowledge base format
            previous_ans_format += qns + '\n'
            previous_ans_format += f"Answer: {ans}" + "\n\n"
            
        # Add instructions for using the knowledge base to answer the question    
        previous_ans_format += "You should use the information above to answer the following question directly and concisely. If the user's question is not related to the knowledge base, answer it directly without using the knowledge base."
    
    else:
        # If no previous answers, set knowledge base format to empty
        previous_ans_format = ""
    
    # Construct the final prompt with knowledge base and user's question
    prompt = f"""
    {previous_ans_format}
    
    Question from the user:
    {query}

    """
    
    # Query the LLM agent with the constructed prompt
    out = llm.query_agent(query=prompt)

    # Return the output of the LLM agent
    return {"agent_out":out}


def multiple_question_parser(state:list):
    """
    Processes and manages multi-part question responses.

    Function Description:
    Handles the state management for multi-part questions by:
    1. Formatting current Q&A pair
    2. Updating answer history
    3. Managing question queue
    4. Preparing next question for processing

    Input:
    - state (dict): Contains:
        - input (str): Current question
        - agent_out: Answer to current question
        - all_answer (list): Previous Q&A history
        - remaining_qns (list): Queue of remaining questions

    Output:
    - dict: Updated state containing:
        - input: Next question or empty string
        - remaining_qns: Updated question queue
        - all_answer: Updated Q&A history

    Note:
    - Returns empty input when all questions processed
    - Maintains numerical ordering of questions in history
    """
    
    # Print the stage prefix for "Multiple Question Parser"
    print(graph_stage_prefix, "Multiple Question Parser")
    
    # Retrieve the current question, agent's output, and all previous answers from the state
    qns = state['input']
    out = state['agent_out']
    all_answer = state["all_answer"]
    
    # Create a dictionary to store the question and answer
    qns_ans_dict = {}
    qns_ans_dict['qns'] = f'{len(all_answer)+1}. {qns}'
    qns_ans_dict['ans'] = out
    
    # Append the current question-answer pair to the list of all answers
    all_answer.append(qns_ans_dict)
    
    # Check if there are any remaining questions
    remaining_qns = state['remaining_qns']
    if remaining_qns:
        # If there are remaining questions, set the next question as the input and update remaining questions
        return {"input":str(remaining_qns[0]), "remaining_qns":remaining_qns[1:], "all_answer":all_answer}
    else:
        # If no remaining questions, return an empty input with all accumulated answers
        return {"input": "", "remaining_qns":[], "all_answer":all_answer}
    
def router_multiple_question(state:list):
    """
    Controls flow for multi-question processing.

    Function Description:
    Determines whether to continue processing questions or end the session
    based on presence of remaining input in the state.

    Input:
    - state (dict): Contains:
        - input (str): Next question to process or empty string
        - remaining_qns (list): Any remaining questions

    Output:
    - str: Next processing stage ('router_agent' or '__end__')

    Note:
    - Returns '__end__' when no more questions to process
    - Maintains processing loop while questions remain
    """
    
    print(graph_stage_prefix, "Multiple Question Router")
    if state["input"]:
        return "router_agent"
    else:
        return "__end__"
    
    