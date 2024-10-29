from langchain_community.llms import Ollama
from regular_agent.agent_prompt import Agent_Prompt

class Agent_Ai:
    def __init__(self, model = "llama3.1", df=[], temperature=0):
        self.llm = Ollama(model=model, temperature=temperature)
        self.df = df
    
    #An llm that answer with a prompt. If the prompt is not given it will use the default prompt
    def prompt_agent(self, query, prompt=""):
        
        '''
        This function is used to invoke a prompt formatted with LangChain PromptTemplate as input
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
    
    #An llm that simply answer the question
    def query_agent(self, query):
        
        '''
        This function is used to invoke a prompt that is of type str as input
        '''
        
        return self.llm.invoke(query)
    
    def run(self, query):
        
        '''
        This funtion format the output in a dictionary type that will be necessary for the app
        '''
        
        return [{"qns":query, "ans" : self.query_agent(query= query)}]