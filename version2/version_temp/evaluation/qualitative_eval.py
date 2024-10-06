import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python_agent.python_ai import Python_Ai
from regular_agent.agent_ai import Agent_Ai
from lang_graph.lang_graph import Graph

import json
import re
import ollama
import csv
from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import os


class LanguageModelEvaluator:
    # Implement the df thing here------------------------------------------------------------
    def __init__(self, model_name: str, model_type: str = 'pandasai'):
        self.model_name = model_name
        self.model_type = model_type
        self.df = [pd.read_csv('./Mac_2k.log_structured.csv')]
        self.model = self._load_model()
        self.graph = self._load_graph()

    def _load_model(self):
        if self.model_type == 'ollama':
            return ollama.Client()
        elif self.model_type == 'pandasai':
            return Python_Ai(df=self.df).pandas_legend()
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported")
    
    def _load_graph(self):
        if self.model_type == 'pandasai' and self.model is not None:
            return Graph(pandas_llm=self.model, df =self.df)
        else:
            raise NotImplementedError(f"Model type {self.model_type} (ie not pandasai) not supported for Graph")

    def generate_response(self, prompt: str, context: str = None) -> str:
        if self.model_type == 'ollama':
            print('ollama being used to answer the prompt')
            response = self.model.chat(model=self.model_name, messages=[
                {
                    'role': 'system',
                    'content': f"You are an AI assistant analyzing log files. Here's the log content:\n\n{context}\n\nAnswer the following question based on the logs provided. Return your answer as a number, not text, at the end where applicable:"
                },
                {
                    'role': 'user',
                    'content': prompt,
                }
            ])
            return response['message']['content']
        elif self.model_type == 'pandasai':
            if self.graph is None:
                raise ValueError("LangGraph is not initialized")
            print('langgraph model being used to answer the prompt')
            return self.graph.run(prompt)
        else:
            raise NotImplementedError(f"Model type {self.model_type} not supported")



    def extract_numeric_value(self, text: str) -> float:
        # Use a non-capturing group to get full matches
        matches = re.findall(r'\d+(?:\.\d+)?', text)
        if matches:
            # Return the last numeric value found
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

    # def read_log_file(self, file_path: str) -> str:
    #     file_path = Path(file_path)
    #     file_extension = file_path.suffix.lower()

    #     if file_extension in ['.txt', '.log']:
    #         with open(file_path, 'r') as f:
    #             return f.read()
    #     elif file_extension == '.csv':
    #         log_content = []
    #         with open(file_path, 'r') as f:
    #             csv_reader = csv.reader(f)
    #             for row in csv_reader:
    #                 log_content.append(','.join(row))
    #         return '\n'.join(log_content)
    #     else:
    #         raise ValueError(f"Unsupported file format: {file_extension}")

    def evaluate(self, log_file: str, ground_truth_file: str, prompts: List[str], metric_names: List[str]) -> Dict[str, float]:
        if self.model_type == 'pandasai':
            df = pd.read_csv(log_file)
            # self.set_pandas_agent(df)
        else:
            log_content = self.read_log_file(log_file)
        
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)

        results = {}
        for prompt, metric_name in zip(prompts, metric_names):
            model_response = self.generate_response(prompt, log_content if self.model_type != 'pandasai' else None)
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
            print(f"Calculated accuracy: {accuracy}%\n")

        return results

    def save_results(self, results: Dict[str, Dict], output_file: str = None):
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"./results/quantitative_evaluation_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        output = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "evaluation_time": datetime.now().isoformat(),
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_file}")
        return output_file

# from language_model_evaluator import LanguageModelEvaluator
def main():
    model_name = "pandasai" 
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

    evaluator = LanguageModelEvaluator(model_name, model_type='pandasai')
    results = evaluator.evaluate(log_file, ground_truth_file, prompts, metric_names)

    print("Evaluation Results:")
    for metric, accuracy in results.items():
        print(f"{metric}: {accuracy:.2f}% accuracy")

if __name__ == "__main__":
    main()