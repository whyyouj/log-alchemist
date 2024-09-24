import pandas as pd
from pandasai import SmartDataframe
import pandasai as pai
from langchain_ollama import OllamaLLM

# LLM
llm = OllamaLLM(
    model = "llama3.1",
    temperature = 0.2
)

# Load Data
df = pd.read_csv("./data/mac/Mac_2k.log_structured.csv")

# Running PandasAI
pandas_ai_agent = SmartDataframe(df, config = {
    "llm" : llm
})

print(pandas_ai_agent.chat("Plot the top 5 EventTemplates."))