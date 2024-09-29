from pandasai import Agent
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from regular_agent.agent_ai import Agent_Ai
from pandasai.responses.streamlit_response import StreamlitResponse

class Python_Ai:
    def __init__(self, model = "codellama:7b", df =None, temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        return Agent_Ai(
            model=self.model, 
            temperature=self.temperature, 
            df=self.df
        )
    
    def pandas_legend(self):
        llm  = self.get_llm().llm
        pandas_ai = Agent(
            self.df, 
            description = """
                You are a data analysis agent tasked with the main goal to answer any data related queries. 
                Everytime I ask you a question, you should provide the code to that specifically answers the question.
            """,
            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":3,
                "response_parser": StreamlitResponse
            }
        )
        return pandas_ai
        

if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv("../../../EDA/data/mac/Mac_2k.log_structured.csv")
    ai = Python_Ai(df=df).pandas_ai_agent('how many users are there and who are the different users')
    print(ai[0].explain(), ai[1])