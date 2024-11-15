from langchain_community.llms import Ollama
from regular_agent.agent_prompt import Agent_Prompt

class Agent_Ai:
    def __init__(self, model = "llama3.1", df=[], temperature=0):
        """
        Initializes an AI agent with specified model parameters and data.

        Function Description:
        Creates an instance of Agent_Ai with a specified language model, dataset, and temperature
        setting for controlling response randomness. Uses Ollama as the underlying LLM framework.

        Input:
        - model (str): The name of the language model to use (default: "llama3.1")
        - df (list): Dataset to be used for analysis or reference (default: empty list)
        - temperature (float): Controls randomness in model outputs (default: 0, most deterministic)

        Output:
        - None: Initializes class attributes

        Note:
        - If no model is specified, defaults to "llama3.1"
        - Empty dataframe initialization is allowed but will limit functionality
        """
        self.llm = Ollama(model=model, temperature=temperature)
        self.df = df
    
    # An LLM that answers with a prompt. If the prompt is not given, it will use the default prompt
    def prompt_agent(self, query, prompt=""):
        """
        Processes queries using a formatted prompt template.

        Function Description:
        Handles user queries by combining them with either a default or custom prompt template.
        Uses LangChain's prompt formatting system for structured interactions with the LLM.

        Input:
        - query (str): The user's question or input
        - prompt (str): Optional custom prompt template (default: empty string)

        Output:
        - str: The model's response to the prompted query

        Note:
        - Returns error message if no dataframe is initialized
        - Uses default Agent_Prompt if no custom prompt is provided
        """
        
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
        """
        Directly queries the language model without prompt formatting.

        Function Description:
        Provides a simplified interface for direct queries to the language model
        without additional prompt engineering or formatting.

        Input:
        - query (str): The raw query to send to the language model

        Output:
        - str: The model's direct response

        Note:
        - No error handling for empty queries
        - Response format depends entirely on the model's output
        """
        return self.llm.invoke(query)
    
    def run(self, query):
        """
        Formats model responses into a standardized dictionary format.

        Function Description:
        Wraps query_agent responses in a structured format suitable for application use,
        maintaining a consistent interface for question-answer pairs.

        Input:
        - query (str): The user's question or input

        Output:
        - list: Contains a dictionary with 'qns' (question) and 'ans' (answer) keys

        Note:
        - Always returns a list with single dictionary, even for empty responses
        - Maintains consistent format for application integration
        """
        
        return [{"qns":query, "ans" : self.query_agent(query= query)}]