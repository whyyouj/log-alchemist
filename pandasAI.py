import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
import pandasai as pai
from langchain_ollama import OllamaLLM

# LLM
llm = OllamaLLM(
    model = "llama3.1"
)

# Load Data
log_df = pd.read_csv("./data/mac/Mac_2k.log_structured.csv")

# Running PandasAI
df = SmartDataframe(log_df, config = {
    "llm" : llm,
    "verbose" : True
})

print(df.chat("Under EventTemplate, list all events with 'ERROR' not just the first few."))
pai.clear_cache()