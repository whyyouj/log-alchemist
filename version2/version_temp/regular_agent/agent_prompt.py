from langchain_core.prompts import PromptTemplate

# Define the prefix and suffix for the prompt template
PREFIX = """
    You are working with one or more pandas dataframe(s) in Python.
    The following shows the first 3 rows of each of the dataframe(s):

    """
SUFFIX = """
    Here is the question from the user:
    {input}

    Your task is to determine if the user question is related to manipulating the above dataframe(s) using Python.
    In particular, can the user question be answered by using Python code to manipulate the dataframe(s)?

    Important:
    - If the question is unrelated to Python data manipulation (like greetings or general questions or general explanation question), answer with a *no*
    Can this be answered by using Python code to manipulate the dataframe?
    
    Answer Should be formatted 
    <Thought> ...
    <Answer> yes or no only.

    """

class Agent_Prompt:
    """
    A class to generate prompt templates that incorporate dataframe previews for AI interactions.
    """
    
    def __init__(self, df):
        """
        Initializes an Agent_Prompt instance with one or more dataframes.

        Function Description:
        Creates a new Agent_Prompt object and sets up the initial prompt components with
        the provided dataframe(s). The prompt will include a preview of each dataframe.

        Input:
        - df (list): A list of pandas DataFrames to be included in the prompt

        Output:
        - None: Initializes instance variables
        
        Note:
        - If df is empty, the prompt will only contain the prefix and suffix without any dataframe previews
        """
        self.df = df
        self.prefix = PREFIX
        self.suffix = SUFFIX
        
    @property
    def value(self):
        """
        Generates and returns a complete prompt template with dataframe previews.

        Function Description:
        Combines the prefix, dataframe previews (first 3 rows of each dataframe in markdown format),
        and suffix into a single prompt template. The template is designed to help an AI model
        understand and respond to questions about dataframe manipulation.

        Input:
        - None: Uses instance variables set during initialization

        Output:
        - template (PromptTemplate): A LangChain prompt template object containing the complete prompt
        
        Note:
        - If no dataframes were provided during initialization, the template will only contain
          the prefix and suffix without any dataframe previews
        - Each dataframe preview is converted to markdown format for better readability
        """
        for d in self.df:
            head = str(d.head(3).to_markdown()) + '\n\n'
            self.prefix += head

        template = PromptTemplate.from_template("\n\n".join([self.prefix, self.suffix]))
        return template