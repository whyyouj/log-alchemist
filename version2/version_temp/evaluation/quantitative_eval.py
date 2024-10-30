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
    def __init__(self):
        self.df = [pd.read_csv('../../../data/Mac_2k.log_structured.csv')]
        

    def generate_response(self, prompt: str) -> str:
        pandas_ai = Python_Ai(df=self.df).pandas_legend()
        # graph = Graph(pandas_ai, self.df)
        query = f"""
        The following is the query from the user:
        {prompt}

        You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

        You are to following the instructions below strictly:
        - Any query related to Date or Time, refer to the 'Datetime' column.
        - Any query related to ERROR, WARNING or EVENT, refer to the EventTemplate column.
        """
        # response = graph.run(prompt)
        response = pandas_ai.chat(query)
        if isinstance(response, bytes):
            return response.decode('utf-8')
        return str(response)

    def extract_numeric_value(self, text: str) -> float:
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        if matches:
            return float(matches[-1])
        else:
            raise ValueError(f"No numeric value found in the text: {text}")

    def evaluate(self, log_file: str, ground_truth_file: str, prompts: List[str], metric_names: List[str], n: int) -> Dict[str, Dict]:
        df = pd.read_csv(log_file)

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
                        self.extract_numeric_value(model_response) == ground_truth_value):
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

def main():
    log_file = "Mac_2k.log_structured.csv"
    ground_truth_file = "mac_ground_truth.json"
    prompts = [
        "How many rows are there in the dataset?", 
        "How many times did the event with eventid E189 occur?",
        "How many times did the event E189 occur?",
        "How many times did the event with component kernel occur?",
        "What is the most frequent eventid that occurred?",
        "What is the total number of errors and warnings are recorded in this log?",
        "What is the most frequent user?",
        "How many missing values are there in Address?",
        "How many times did the event with component kernel occur?",
        "How many times did the event with the user authorMacBook-Pro occur?",

        # "How many times did the event with eventid E120 occur?",
        # "How many times did the event with eventid E203 occur?",
        # "How many times did the event with eventid E323 occur?",
        # "How many times did the event with component com.apple.cts occur?",
        # "How many times did the event with component corecaptured occur?",
        # "How many times did the event with component QQ occur?",
        # "How many times did the event with component Microsoft Word occur?",
        
    ]
    metric_names = ["total_rows", 'E189', "E189", "kernel", "most_frequent_eventid",
                     "errors_warnings", "most_freq_user", 'missing_val_address', 'kernel', 'authorMacBook-Pro']
                    # , 'E120', 'E203', 'E323', 'kernel', 'com.apple.cts', 'corecaptured', 'QQ', 'Microsoft Word', 'authorMacBook-Pro']

    n = int(input("Enter the number of times to run each evaluation: "))

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names, n)

    evaluator.save_results(results)

    return results

if __name__ == "__main__":
    evaluation_results = main()
    
global_graph = None