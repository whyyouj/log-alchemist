from langchain_community.llms import Ollama
from regular_agent.agent_prompt import Agent_Prompt

class Agent_Ai:
    def __init__(self, model = "llama3.1", df=[], temperature=0):
        '''
        Description: Initializes the Agent_Ai object with a model, dataframe, and temperature.
        
        Input:
        - model: str (default: "llama3.1")
        - df: list (optional)
        - temperature: float (optional)
        
        Output: None
        '''
        self.llm = Ollama(model=model, temperature=temperature)
        self.df = df
    
    # An LLM that answers with a prompt. If the prompt is not given, it will use the default prompt
    def prompt_agent(self, query, prompt=""):
        '''
        Description: Invokes a prompt formatted with LangChain PromptTemplate as input.
        
        Input:
        - query: str
        - prompt: str (optional)
        
        Output:
        - agent_out: str
        '''
        
        if len(self.df) == 0:
            return 'Error: No dataframe found'
        
        if prompt == "":
            prompt = Agent_Prompt(self.df).value
            agent_out = prompt | self.llm
            return agent_out.invoke({"input" : query})
        
        else:
            agent_out = prompt | self.llm
            return agent_out.invoke({"input": query})
    
    # An LLM that simply answers the question
    def query_agent(self, query):
        '''
        Description: Invokes a prompt that is of type str as input.
        
        Input:
        - query: str
        
        Output:
        - response: str
        '''
        
        return self.llm.invoke(query)
    
    def run(self, query):
        '''
        Description: Formats the output in a dictionary type that will be necessary for the app.
        
        Input:
        - query: str
        
        Output:
        - result: list of dict
        '''
        
        return [{"qns":query, "ans" : self.query_agent(query= query)}]