from pandasai.llm.local_llm import LocalLLM
from langchain_community.llms import Ollama
import streamlit as st
from pandasai.connectors import PostgreSQLConnector
from pandasai import SmartDataframe
my_connector = PostgreSQLConnector(
    config={
        "host":"localhost",
        "port":5432,
        "database":"llm",
        "username":"chuapk",
        "table" : "flights",
        "password" : "",
    }
)



model = Ollama(model='llama3-groq-tool-use')
df_connector = SmartDataframe(my_connector, config={"llm":model})

st.title("MySQL with Llama-3")
prompt = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        st.spinner("Generating response...")
        st.write(df_connector.chat(prompt))


