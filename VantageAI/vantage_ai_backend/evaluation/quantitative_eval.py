import json
import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import time
import sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lang_graph.lang_graph import Graph
from python_agent.python_ai import Python_Ai

class LanguageModelEvaluator:
    """
    A class for evaluating language model responses against ground truth for log analysis tasks.
    """
    def __init__(self):
        # self.df = [pd.read_csv('../../../data/Mac_2k.log_structured.csv')]
        # self.df = [pd.read_csv("Windows_2k.log_structured.csv")]
        self.df = [pd.read_csv("auditrecords.csv")]

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response using a language model for a given prompt about log analysis.

        Function Description:
        Uses a specialized pandas-aware language model to interpret and answer queries about log data.
        The model is instructed to return executable code without function definitions.

        Input:
        - prompt (str): The user query about log data analysis

        Output:
        - str: The model's response, decoded if in bytes format

        Note:
        - Returns empty string if the model fails to generate a response
        """
        # PANDAS_LLM = 'jiayuan1/llm2'
        PANDAS_LLM = 'jiayuan1/pandas-instruct-30'
        # PANDAS_LLM = "llama3.1"
        pandas_ai = Python_Ai(PANDAS_LLM, df=self.df).pandas_legend_with_skill()
        # graph = Graph(pandas_ai, self.df)
        query = f"""
        The following is the query from the user:
        {prompt}

        You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

        You are to following the instructions below strictly:
        - Any query related to Date or Time, refer to the 'Datetime' column.
        """
        
        response = pandas_ai.chat(prompt)
        if isinstance(response, bytes):
            return response.decode('utf-8')
        return str(response)

    def extract_numeric_value(self, text: str) -> float:
        """
        Extracts the last numeric value from a text string.

        Function Description:
        Uses regex to find all numbers (including decimals) in the text and returns the last one found.

        Input:
        - text (str): Text containing numeric values

        Output:
        - float: The last numeric value found in the text

        Note:
        - Raises ValueError if no numeric value is found
        """
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        if matches:
            return float(matches[-1])
        else:
            raise ValueError(f"No numeric value found in the text: {text}")

    def evaluate(self, log_file: str, ground_truth_file: str, prompts: List[str], metric_names: List[str], n: int) -> Dict[str, Dict]:
        """
        Evaluates model performance by comparing responses against ground truth values.

        Function Description:
        Runs multiple evaluations for each prompt, comparing model responses with expected values.
        Tracks accuracy, timing, and stores all responses for analysis.

        Input:
        - log_file (str): Path to the structured log CSV file
        - ground_truth_file (str): Path to JSON file containing correct answers
        - prompts (List[str]): List of questions to ask the model
        - metric_names (List[str]): Corresponding metric names for ground truth lookup
        - n (int): Number of times to run each evaluation

        Output:
        - Dict[str, Dict]: Detailed results including accuracy, timing, and responses

        Note:
        - Returns empty dictionary if evaluation fails
        """
        self.df = pd.read_csv(log_file)

        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        results = {}
        overall_correct = 0
        overall_total = n * len(prompts)
        overall_time = 0
        counter = 1

        for prompt, metric_name in zip(prompts, metric_names):
            correct_predictions = 0
            responses = []
            start_time = time.time()

            for _ in range(n):
                model_response = self.generate_response(prompt)
                responses.append(model_response)
                
                try:
                    ground_truth_value = ground_truth[metric_name]
                    
                    if (str(ground_truth_value) in model_response or
                        str(model_response) in str(ground_truth_value)):
                        correct_predictions += 1
                        overall_correct += 1
                    
                except ValueError as e:
                    print(f"Error processing {metric_name}: {str(e)}")

            end_time = time.time()
            total_time = end_time - start_time
            overall_time += total_time

            accuracy = (correct_predictions / n) * 100

            results[f"Question {counter}"] = {
                "prompt": prompt,
                "model_responses": responses,
                "ground_truth_value": ground_truth[metric_name],
                "correct_predictions": correct_predictions,
                "total_runs": n,
                "accuracy": accuracy,
                "total_time": total_time
            }
            counter += 1

            print(f"{metric_name}: {accuracy:.2f}% accuracy ({correct_predictions}/{n} correct), Time: {total_time:.2f} seconds")

        overall_accuracy = (overall_correct / overall_total) * 100
        results["overall"] = {
            "correct_predictions": overall_correct,
            "total_runs": overall_total,
            "accuracy": overall_accuracy,
            "total_time": overall_time
        }

        print(f"\nOverall accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total} correct)")
        print(f"Total time: {overall_time:.2f} seconds")

        return results

    def save_results(self, results: Dict[str, Dict], output_file: str = None):
        """
        Saves evaluation results to a JSON file.

        Function Description:
        Creates a timestamped JSON file containing evaluation results and metadata.

        Input:
        - results (Dict[str, Dict]): Evaluation results to save
        - output_file (str, optional): Custom output file path

        Output:
        - str: Path to the saved results file

        Note:
        - Creates default filename with timestamp if output_file is None
        """

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"./results/quantitative_evaluation_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output = {
            "evaluation_time": datetime.now().isoformat(),
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_file}")
        return output_file

def run_train_evaluation():
    """
    Runs evaluation on Mac log dataset with predefined prompts.

    Function Description:
    Executes evaluation using Mac-specific log data and corresponding ground truth values.
    Uses a set of predefined prompts focused on Mac log analysis.

    Input:
    None

    Output:
    - Dict[str, Dict]: Evaluation results

    Note:
    - Prompts user for number of evaluation runs
    - Returns None if evaluation fails
    """
    log_file = "Mac_2k.log_structured.csv"
    ground_truth_file = "mac_ground_truth.json"
    prompts = [
        "How many rows are there in the dataset?", 
        "How many times did the event with eventid E189 occur?",
        "How many times did the event E189 occur?",
        "How many times did the event with component kernel occur?",
        "What is the most frequent eventid that occurred?",
        "Who is the top user?",
        "What is the total number of errors and warnings are recorded in this log?",
        "How many missing values are there in Address?",
        "How many authorMacBook-Pro user are there?",
        "What is the total number of times that eventid E189 and E188 occur?"
    ]
    metric_names = ["total_rows", 'E189', "E189", "kernel", "most_frequent_eventid", 
                    "most_freq_user", "errors_warnings", 'missing_val_address', 'authorMacBook-Pro', "e189_e188_sum"]
    
    n = int(input("Enter the number of times to run each evaluation: "))

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names, n)

    evaluator.save_results(results)
    return results

def run_test_evaluation():
    """
    Runs evaluation on Windows log dataset with predefined prompts.

    Function Description:
    Executes evaluation using Windows-specific log data and corresponding ground truth values.
    Uses a set of predefined prompts focused on Windows log analysis.

    Input:
    None

    Output:
    - Dict[str, Dict]: Evaluation results

    Note:
    - Prompts user for number of evaluation runs
    - Returns None if evaluation fails
    """
    log_file = "Windows_2k.log_structured.csv"
    ground_truth_file = "windows_ground_truth.json"
    prompts = [
        "How many rows are there in the dataset?", 
        "How many times did the event with eventid E29 occur?",
        "How many times did the event E29 occur?",
        "How many times did the event with component CBS occur?",
        "What is the most frequent eventid that occurred?",
        "What is the number of errors recorded?",
        "What is the number of warnings recorded?",
        "Who is the top EventTemplate?",
        "How many missing values are there in Content?",
        "How many 'Warning: Unrecognized packageExtended attribute.' are there?",
        "What is the total number of times that eventid E29 and E36 occur?"
    ]
    metric_names = ["total_rows", 'E29', "E29", "CBS", "most_freqent_eventid",
                     "errors", "warnings", "top_event_template", 'missing_val_content', 'warning_unrecog', "e29_e36_total"]
    
    n = int(input("Enter the number of times to run each evaluation: "))

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names, n)

    evaluator.save_results(results)
    return results

def run_test_evaluation_audit():
    """
    Runs evaluation on Audit log dataset with predefined prompts.

    Function Description:
    Executes evaluation using audit-specific log data and corresponding ground truth values.
    Uses a set of predefined prompts focused on audit log analysis.

    Input:
    None

    Output:
    - Dict[str, Dict]: Evaluation results

    Note:
    - Prompts user for number of evaluation runs
    - Returns None if evaluation fails
    """
    log_file = "auditrecords.csv"
    ground_truth_file = "auditrecords_ground_truth.json"
    prompts = [
        "How many rows are there in the dataset?", 
        "What is the number of ExchangeAdmin in RecordType?",
        "How many times did the Identity ae99738a-9771-4f58-625e-08d94ade678f occur?",
        "Who is the top Operations?",
        "Who is the top RecordType?",
        "How many distinct values does AuditData have?",
        "How many missing values are there in Identity?",
        "What is the total number of rows of ExchangeAdmin and AzureActiveDirectoryStsLogon in RecordType?"
    ]
    metric_names = ["total_rows", "exchangeAdmin", "identity_uniq", "top_operations",
                    "top_recordType", "auditDataDistinct", "missing_identity", "total_num_agg"]
    
    n = int(input("Enter the number of times to run each evaluation: "))

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names, n)

    evaluator.save_results(results)
    return results

def main():
    # run_train_evaluation()
    # run_test_evaluation()
    run_test_evaluation_audit()

main()