import json
import re
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
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

    def calculate_accuracy(self, predicted: float, actual: float) -> float:
        if predicted == actual:
            return 100.0
        elif actual == 0:
            return 0.0 if predicted != 0 else 100.0
        else:
            error = abs(predicted - actual) / actual
            accuracy = max(0, (1 - error)) * 100
            return min(accuracy, 100.0)

    def evaluate(self, log_file: str, ground_truth_file: str, prompts: List[str], metric_names: List[str]) -> Dict[str, Dict]:
        df = pd.read_csv(log_file)

        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        results = {}
        for prompt, metric_name in zip(prompts, metric_names):
            model_response = self.generate_response(prompt)
            print(f"Prompt: {prompt}")
            print(f"Model response: {model_response}")
            try:
                extracted_value = self.extract_numeric_value(model_response)
                ground_truth_value = ground_truth[metric_name]
                print(f"Extracted value: {extracted_value}")
                print(f"Ground truth value: {ground_truth_value}")
                accuracy = self.calculate_accuracy(extracted_value, ground_truth_value)
                results[metric_name] = {
                    "prompt": prompt,
                    "model_response": model_response,
                    "extracted_value": extracted_value,
                    "ground_truth_value": ground_truth_value,
                    "accuracy": accuracy
                }
            except ValueError as e:
                print(f"Error processing {metric_name}: {str(e)}")
                results[metric_name] = {
                    "prompt": prompt,
                    "model_response": model_response,
                    "error": str(e),
                    "accuracy": 0.0
                }
            print(f"Calculated accuracy: {results[metric_name]['accuracy']}%\n")

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
        "How many times did the event with eventid E188 occur?",
        "How many times did the event with eventid E120 occur?",
        "How many times did the event with eventid E203 occur?",
        "How many times did the event with eventid E323 occur?",
        "How many times did the event with component kernel occur?",
        "How many times did the event with component com.apple.cts occur?",
        "How many times did the event with component corecaptured occur?",
        "How many times did the event with component QQ occur?",
        "How many times did the event with component Microsoft Word occur?",
        "How many times did the event with the user authorMacBook-Pro occur?",
    ]
    metric_names = ['E189', 'E188', 'E120', 'E203', 'E323', 'kernel', 'com.apple.cts', 'corecaptured', 'QQ', 'Microsoft Word', 'authorMacBook-Pro']

    evaluator = LanguageModelEvaluator()
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names)

    print("Evaluation Results:")
    for metric, result in results.items():
        print(f"{metric}: {result['accuracy']:.2f}% accuracy")

    evaluator.save_results(results)

    return results

if __name__ == "__main__":
    evaluation_results = main()