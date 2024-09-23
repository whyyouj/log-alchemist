import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from langchain_ollama import OllamaLLM

# LLM
llm = OllamaLLM(
    model = "llama3.1"
)

# st.title("Log Data Analysis")

# uploader_file = st.file_uploader("Upload a file", type = ["csv"])

# if uploader_file is not None:
#     data = pd.read_csv(uploader_file)
#     st.write(data.head(5))
#     df = SmartDataframe(data, config = {"llm" : llm})
#     prompt = st.text_area("Enter your prompt:")

#     if st.button("Generate"):
#         if prompt:
#             with st.spinner("Generating response..."):
#                 st.write(df.chat(prompt))
#         else:
#             st.warning("Please enter a prompt!")


# Load Data
log_df = pd.read_csv("./data/mac/Mac_2k.log_structured.csv")

# Running PandasAI
df = SmartDataframe(log_df, config = {"llm" : llm})
print(df.chat("Summarise the log data."))

