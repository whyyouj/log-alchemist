import json
import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
import time
from graph_manager import Graph, global_graph

class LanguageModelEvaluator:
    def __init__(self):
        if global_graph is None:
            self.graph = Graph.create_graph()
        else:
            self.graph = global_graph

    def generate_response(self, prompt: str) -> str:
        response = self.graph.run(prompt)
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
        overall_total = 0
        overall_time = 0

        for prompt, metric_name in zip(prompts, metric_names):
            correct_predictions = 0
            responses = []
            start_time = time.time()

            for _ in range(n):
                model_response = self.generate_response(prompt)
                responses.append(model_response)
                
                try:
                    extracted_value = self.extract_numeric_value(model_response)
                    ground_truth_value = ground_truth[metric_name]
                    
                    if extracted_value == ground_truth_value:
                        correct_predictions += 1
                        overall_correct += 1
                    
                    overall_total += 1
                except ValueError as e:
                    print(f"Error processing {metric_name}: {str(e)}")

            end_time = time.time()
            total_time = end_time - start_time
            overall_time += total_time

            accuracy = (correct_predictions / n) * 100

            results[metric_name] = {
                "prompt": prompt,
                "model_responses": responses,
                "ground_truth_value": ground_truth[metric_name],
                "correct_predictions": correct_predictions,
                "total_runs": n,
                "accuracy": accuracy,
                "total_time": total_time
            }

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
        "How many times did the event with eventid E189 occur?",
        "How many times did the event with eventid E188 occur?"
        # "How many times did the event with eventid E120 occur?",
        # "How many times did the event with eventid E203 occur?",
        # "How many times did the event with eventid E323 occur?",
        # "How many times did the event with component kernel occur?",
        # "How many times did the event with component com.apple.cts occur?",
        # "How many times did the event with component corecaptured occur?",
        # "How many times did the event with component QQ occur?",
        # "How many times did the event with component Microsoft Word occur?",
        # "How many times did the event with the user authorMacBook-Pro occur?",
    ]
    metric_names = ['E189', 'E188']
    #, 'E120', 'E203', 'E323', 'kernel', 'com.apple.cts', 'corecaptured', 'QQ', 'Microsoft Word', 'authorMacBook-Pro']

    n = int(input("Enter the number of times to run each evaluation: "))

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names, n)

    evaluator.save_results(results)

    return results

if __name__ == "__main__":
    evaluation_results = main()