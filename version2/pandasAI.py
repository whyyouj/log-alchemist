import pandas as pd
from pandasai import SmartDataframe
import pandasai as pai
from langchain_ollama import OllamaLLM

# LLM
llm = OllamaLLM(
    model = "llama3.1"
)

# Load Data
df = pd.read_csv("./data/mac/Mac_2k.log_structured.csv")
df["Combined"] = df["Month"] + " " + df["Date"].astype(str) + " " + df["Time"]
df["Datetime"] = pd.to_datetime(df["Combined"], format='%b %d %H:%M:%S')
df.drop(columns = ["Date", "Month", "Time"], inplace = True)

# Running PandasAI
pandas_ai_agent = SmartDataframe(df, config = {
    "llm" : llm,
    "enable_cache" : False,
    "max_retries": 5
})

print(pandas_ai_agent.chat("Are there any anomalies?"))