from langchain_core.prompts import PromptTemplate

PREFIX = """
    You are working with one or more pandas dataframe(s) in Python.
    The following shows the first 5 rows of each of the dataframe(s):

    """
SUFFIX = """
    Here is the question from the user:
    {input}

    Your task is to determine if the user question is related to manipulating the above dataframe(s) using Python.
    In particular, can the user question be answered by using Python code to manipulate the dataframe(s)?

    Important:
    - Answer with a *yes* or a *no* only.
    - DO NOT supplement your answer with anything else.
    - If the question is unrelated to Python data manipulation (like greetings or general questions), answer with a *no*
    """
class Agent_Prompt:
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
        # df_head = str(self.df.head(3).to_markdown())
        # template = template.partial(df_head = df_head)
        return template