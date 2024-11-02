from langchain_core.prompts import PromptTemplate

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
    
    '''
    This class output a prompt template that incooperates a dataframe
    '''
    
    def __init__(self, df):
        self.df = df
        self.prefix = PREFIX
        self.suffix = SUFFIX
        
    @property
    def value(self):
        for d in self.df:
            head = str(d.head(3).to_markdown()) + '\n\n'
            self.prefix += head

        template = PromptTemplate.from_template("\n\n".join([self.prefix, self.suffix]))
        return template