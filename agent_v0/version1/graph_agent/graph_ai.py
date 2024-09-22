import pandas as pd
import numpy as np
from graph_agent.graph_prompt import Graph_Prompt
from langchain_community.llms import Ollama
from langchain_experimental.utilities import PythonREPL
import re
from python_agent.python_ai import Python_Ai

class Graph_Ai:
    def __init__(self,model = "llama3", temperature = 0, df = None, *args, **kwargs):
        self.model = model
        self.temperature = temperature
        self.df  = df
        self.x, self.y, self.z = None, None, None
        
        for expected_k in ['x', 'y', 'z','df', 'llm']:
            if expected_k in kwargs:
                setattr(self, expected_k, kwargs[expected_k])
            
        if self.df is None:
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    self.df = arg
                    break
               
                
    def run(self, prompt):
        p = Graph_Prompt(prompt, self.df, self.x, self.y, self.z)
        llm = Python_Ai(model=self.model, temperature=self.temperature, df=self.df)
        out, input_code = llm.code_executor_agent(p.value)
        return out
    
    def test(self):
        print("hello world")
        