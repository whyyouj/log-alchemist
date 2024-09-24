
from langchain_core.prompts import PromptTemplate

PREFIX = """
    You are working with a pandas dataframe in Python. The name of the dataframe is `df`
    This is the result of `print(df.head())`:
    {df_head}."""
SUFFIX = """
    Here is the Qustion:
    {input}
    Does this question require the manipulating the dataframe above.
    """
class Agent_Prompt:
    def __init__(self, df):
        self.df = df
        self.prefix = PREFIX
        self.suffix = SUFFIX
        
    @property
    def value(self):
        template = PromptTemplate.from_template("\n\n".join([self.prefix, self.suffix]))
        df_head = str(self.df.head(3).to_markdown())
        template = template.partial(df_head = df_head)
        return template
        