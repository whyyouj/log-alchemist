from pandasai import Agent
from regular_agent.agent_ai import Agent_Ai
from pandasai.responses.streamlit_response import StreamlitResponse

class Python_Ai:
    def __init__(self, model = "llama3.1", df =None, temperature=0):
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        return Agent_Ai(model=self.model, temperature=self.temperature, df=self.df)
    
    # def pandas_ai_agent(self, query):
    #     llm = self.get_llm().llm
    #     pandas_ai = Agent(self.df, config={
    #         "llm":llm,
    #         "open_charts":False,
    #         "enable_cache" : False,
    #         "save_charts": True,
    #         "max_retries":1,
    #         "response_parser": StreamlitResponse
    #     })
    #     return pandas_ai.chat(query)
    
    def pandas_legend(self):
        llm  = self.get_llm().llm
        pandas_ai = Agent(self.df, config={
            "llm":llm,
            "open_charts":False,
            "enable_cache" : False,
            "save_charts": True,
            "max_retries":3,
            "response_parser": StreamlitResponse
        })
        return pandas_ai
        

if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv("../../../EDA/data/mac/Mac_2k.log_structured.csv")
    ai = Python_Ai(df=df).pandas_ai_agent('how many users are there and who are the different users')
    print(ai[0].explain(), ai[1])