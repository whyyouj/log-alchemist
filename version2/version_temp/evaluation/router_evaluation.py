# Custom dataset will be made and used, router will be tested to see if routing can be done properly
import json
import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import time
import logging
import  sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lang_graph.lang_graph_utils import  router_agent




def run_router_agent(query, df):
    """
    Processes a query using a router agent to determine appropriate responses based on the provided data.

    Function Description:
    This function takes a user query and a DataFrame containing log data, creates a state dictionary 
    with these inputs, and passes them to a router agent. The router agent then determines the 
    appropriate response based on the query's content and the available data.

    Input:
    - query (str): The user's question or query about the log data
    - df (pandas.DataFrame): DataFrame containing the log data to be analyzed

    Output:
    - result (dict): Contains the router agent's response with a key 'agent_out' holding the answer
    
    Note:
    - If the router_agent fails to process the query, it may return an error message or None
    """

    state = {
        "input": query,  
        "df": df 
    }

    # Change this when you import a diff function from langgraph_utils
    result = router_agent(state)
    return result



# Dataframe being used
data_df = [pd.read_csv('../../../data/Mac_2k.log_structured.csv')]



# Logging
log_file = 'routing_eval.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Marker to indicate start of a evaluation run
logging.info("===================================================")

# Load the Excel file with queries and answers
# If you have a different set of queries to test, change the path here!
excel_file_path = 'router_testing.csv'  
df = pd.read_csv(excel_file_path)

# Tracking correct answers and the total number of queries
correct_answers = 0
total_queries = len(df)

# Start the clock!!
start_time = time.time()

# Iterate through each query in the Excel file
for index, row in df.iterrows():
    query = row['query']
    expected_answer = row['answer']
    
    # Passing query to model and getting the result
    result = run_router_agent(query, data_df)
    
    # Check if the result contains "Answer: yes" or "Answer: no"
    model_answer = result["agent_out"]
    
    # Log the query, expected answer
    logging.info(f"Query: {query}")
    logging.info(f"Expected Answer: {expected_answer}")
    # logging.info(f"Model Answer: {model_answer}")
    
    
    # Compare the model's answer with the expected answer
    if "Answer: yes" in model_answer:
        logging.info(f"Model Answer: yes")
        if expected_answer.strip().lower() == "yes":
            correct_answers += 1
        
    elif "Answer: no" in model_answer:
        logging.info(f"Model Answer: no")
        if expected_answer.strip().lower() == "no":
            correct_answers += 1
    
    logging.info(f"Current Correct Answers: {correct_answers} \n")

        
        


# Calculate the total time taken
end_time = time.time()
time_taken = end_time - start_time

# Log the final result
logging.info(f"Total Correct Answers: {correct_answers}/{total_queries}")
logging.info(f"Time Taken: {time_taken:.2f} seconds")
logging.info("=================================================== \n")

# Display the final results
final_result = {
    "Total Queries": total_queries,
    "Correct Answers": correct_answers,
    "Time Taken (seconds)": time_taken
}

print("Final Results:")
print(final_result)


# Potential work
    # Better arrange file structure