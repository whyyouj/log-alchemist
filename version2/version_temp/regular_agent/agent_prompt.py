from langchain_core.prompts import PromptTemplate

PREFIX = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`
    This is the result of head of the data:
    {df_head}.
    Your task is to determine if the following question is related to manipulating the dataframe using Python.
    If the question is unrelated to Python data manipulation (like greetings or general questions), answer 'no'."""
SUFFIX = """
    Here is the Question:
    {input}
    Can this question be answered by using Python code to manipulate the dataframe?

    Important:
    - Answer with a *yes* or a *no* only.
    - DO NOT supplement your answer with anything else.
    """
class Agent_Prompt:
    def __init__(self, df):
        self.df = df
        self.prefix = PREFIX
        self.suffix = SUFFIX
        
    @property
    def value(self):
        template = PromptTemplate.from_template("\n\n".join([self.prefix, self.suffix]))
        df_head = str(self.df.head(3).to_json())
        template = template.partial(df_head = df_head)
        return template