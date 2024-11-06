import os
import sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import logging
from regular_agent.agent_ai import Agent_Ai

# Configuration of logs
logging.basicConfig(
    filename="multi_question_agent.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Train data
train_data = pd.read_csv("train_router_3.csv")

# Test data
test_inputs = [
    "Filter rows where Component is memory and compare with Component values of disk.",
    "Summarize the dataset by counting unique entries in the Hostname column.",
    "Group data by Application and EventType, then compare entry counts per group.",
    "Retrieve entries with EventId C452 and calculate average Duration for these entries.",
    "Summarize the occurrences of each EventType across different User values.",
    "List all unique IP addresses associated with EventType 'alert' and tell me what is the capital of India.",
    "Filter rows by Timestamp within the range '2024-01-01' to '2024-12-31'.",
    "Count the number of entries where Component is 'network' and EventId is A432.",
    "Summarize data by calculating the total Duration per Component.",
    "Retrieve records where Component is 'security' and EventType is 'error'.",
    "Filter the data to only include rows where User is 'admin' and summarize their entries by EventType.",
    "Compare the number of entries per EventId for the Application 'system_logger'.",
    "Calculate the average Duration for each unique User entry and give me the anomalies in the data set.",
    "Group entries by Component and Hostname, then count entries per group.",
    "Filter rows where IP address starts with '192.' and group by EventType.",
    "Count the number of entries where EventType is 'warning' and Component is 'disk'.",
    "Retrieve and list unique User values associated with EventType 'critical'.",
    "Compare the number of entries across different Components where Timestamp falls within 2024. Give me a summary of the data set",
    "Filter rows with EventId starting with 'D' and summarize by Application name.",
    "Group data by Timestamp (by month) and calculate the total Duration for each month.",
    "What is the meaning of life",
    "Filter for User then count it.",
    "Filter for User then count it. What is the meaning of life. Can you give me the number of rows of data.",
    "Can you give me the number of rows of data?",
    "Give me the table for the highest number of products by the top five countries and explain the results.",
    "Filter the dataset for event that occurred on 2024, only include rows when Component is kernel and return the filtered dataset",
    "Filter the rows where content contains the word error, filter for events that occurred after 2024 and return that filtered dataset.",
    "Extract the row where PID is 2 in 2024 and give me the summary of the entire dataset",
    "Filter where Component is kernel and plot a time series of the number of rows of data per minute.",
    "Tell me about the meaning of life. Calculate the mean, median and standard deviation of the number of rows of data per minute. Explain your results."
]

test_outputs = [
    "[{“Pandas”: “Filter rows where Component is memory and compare with Component values of disk.”}]",
    "[{“Pandas”: “Summarize the dataset by counting unique entries in the Hostname column.”}]",
    "[{“Pandas”: “Group data by Application and EventType, then compare entry counts per group.”}]",
    "[{“Pandas”: “Retrieve entries with EventId C452 and calculate average Duration for these entries.”}]",
    "[{“Pandas”: “Summarize the occurrences of each EventType across different User values.”}]",
    "[{“Pandas”: “List all unique IP addresses associated with EventType ‘alert’.”}, {“General”: “Tell me what is the capital of India?”}]",
    "[{“Pandas”: “Filter rows by Timestamp within the range ‘2024-01-01’ to ‘2024-12-31’.”}]",
    "[{“Pandas”: “Count the number of entries where Component is ‘network’ and EventId is A432.”}]",
    "[{“Pandas”: “Summarize data by calculating the total Duration per Component.”}]",
    "[{“Pandas”: “Retrieve records where Component is ‘security’ and EventId is ‘error’.”}]",
    "[{“Pandas”: “Filter the data to only include rows where User is ‘admin’ and summarize their entries by EventType.”}]",
    "[{“Pandas”: “Compare the number of entries per EventId for the Application ‘system_logger’.”}]",
    "[{“Pandas”: “Calculate the average Duration for each unique User entry.”}, {“Pandas”: “Give me the anomalies in the data set.”}]",
    "[{“Pandas”: “Group entries by Component and Hostname, then count entries per group.”}]",
    "[{“Pandas”: “Filter rows where IP address starts with ‘192.’ and group by EventType.”}]",
    "[{“Pandas”: “Count the number of entries where EventId is ‘warning’ and Component is ‘disk’.”}]",
    "[{“Pandas”: “Retrieve and list unique User values associated with EventType ‘critical’.”}]",
    "[{“Pandas”: “Compare the number of entries across different Components where Timestamp falls within 2024.”}, {“Pandas”: “Give me a summary of the data set.”}]",
    "[{“Pandas”: “Filter rows with EventId starting with ‘D’ and summarize by Application name.”}]",
    "[{“Pandas”: “Group data by Timestamp (by month) and calculate the total Duration for each month.”}]",
    '[{"General": "What is the meaning of life?"}]',
    '[{“Pandas”: “Filter for User then count it.”}]',
    '[{“Pandas”: “Filter for User then count it.”}, {“General”: “What is the meaning of life.”}, {“Pandas”: “Can you give me the number of rows of data.”}]',
    '[{“Pandas”: “Give me the number of rows of data.”}]',
    '[{“Pandas”: “Give me the table for the highest number of products by the top five countries.”}, {“Explain”: “Explain the results.”}]',
    '[{"Pandas": "Filter the dataset for events that occurred on 2024, only include rows when Component is kernel and return the filtered dataset."}]',
    '[{“Pandas”: “Filter the rows where content contains the word error, filter for events that occurred after 2024 and return that filtered dataset.”}]',
    '[{"Pandas": "Extract the row where PID is 2 in 2024."}, {"Pandas": "Give me the summary of the entire dataset."}]',
    '[{“Pandas”: “Filter where Component is kernel and plot a time series of the number of rows of data per minute.”}]',
    '[{“General”: “Tell me about the meaning of life.”}, {“Pandas”: “Calculate the mean, median and standard deviation of the number of rows of data per minute.”}, {“Explain”: “Explain your results.”}]'
]

test_inputs_df = pd.DataFrame(test_inputs, columns=["Input"])
test_inputs_df["Output"] = test_outputs

# Function to generate LLM responses
def generate_response(query):
    llm = Agent_Ai("jiayuan1/router_160", temperature = 0)
    output = llm.query_agent(query)
    return output

# Evaluation function
def evaluate(data):
    queries = data["Input"]
    ground_truths = data["Output"]    

    correct_counts = 0
    wrong_counts = 0
    total_counts = 0
    
    logging.info("=== Starting Evaluation Process ===")

    for i in range(len(queries)):
        total_counts += 1
        logging.info(f"Evaluating Query {i + 1}/{len(queries)}")
        logging.info(f"Query: {queries[i]}")
        try:
            response = generate_response(queries[i])
            logging.info(f"Response: {response}")
            if response in ground_truths[i] or ground_truths[i] in response:
                logging.info("Result: Passed\n")
                correct_counts += 1
            else:
                logging.warning("Result: Failed")
                logging.warning(f"Correct Response: {ground_truths[i]}\n")
                wrong_counts += 1

        except Exception as e:
            logging.error(f"Error processing Query {i + 1}: {e}")
            wrong_counts += 1
            logging.error("Result: Failed due to error\n")
    
    # Summary logging
    accuracy = round(correct_counts / total_counts, 3) if total_counts > 0 else 0
    logging.info("=== Evaluation Summary ===")
    logging.info(f"Total Queries Evaluated: {total_counts}")
    logging.info(f"Passed: {correct_counts}")
    logging.info(f"Failed: {wrong_counts}")
    logging.info(f"Accuracy: {accuracy:.3f}")
    logging.info("===================================\n")

    # Print summary to console
    print(f"Passed: {correct_counts}")
    print(f"Failed: {wrong_counts}")
    print(f"Accuracy: {accuracy:.3f}")
            
if __name__ == "__main__":
    # evaluate(train_data)
    evaluate(test_inputs_df)