import pandas as pd
import matplotlib.pyplot as plt
import logging


logging.basicConfig(filename='code_snippets_test.log', level=logging.INFO)

# Load the CSV file containing log data
data_file = 'Sales Transaction v.4a.csv'
dfs = [pd.read_csv(data_file)]

# Load the Excel file containing code snippets (dataset)
code_snippets_file = 'train_data.xlsx'
snippets_df = pd.read_excel(code_snippets_file)


total_snippets = len(snippets_df)
passed_snippets = 0
failed_snippets = 0


for index, row in snippets_df.iterrows():
    input_query = row['Input']
    code_to_test = row['Output']

    try:
        local_scope = {}

        # This executes the code snippet
        exec(code_to_test, globals(), local_scope)

        # Check if the "result" exists in the local scope after execution
        if 'result' in local_scope:
            print(f"Query {index + 1}: {input_query}")
            print(f"Result: {local_scope['result']}")
            print("\n")
            logging.info(f"Query {index + 1} passed: {input_query}")
            passed_snippets += 1
        else:
            print(f"Query {index + 1}: {input_query}")
            print("Result not found.")
            logging.warning(f"Query {index + 1} failed (no result): {input_query}")
            failed_snippets += 1

    except Exception as e:
        # Log the error and continue with the next snippet
        print(f"Error in Query {index + 1}: {input_query}")
        print(f"Error: {e}\n")
        logging.error(f"Error in Query {index + 1}: {input_query}. Error: {e}")
        failed_snippets += 1

# Prints the  summary of results
print(f"Total code snippets: {total_snippets}")
print(f"Passed: {passed_snippets}")
print(f"Failed: {failed_snippets}")

# Logging the  summary to the log file
logging.info(f"Total code snippets: {total_snippets}")
logging.info(f"Passed: {passed_snippets}")
logging.info(f"Failed: {failed_snippets}")


print("Log saved to 'code_snippets_test.log'")
