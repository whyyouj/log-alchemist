import streamlit as st
from langchain_community.llms import Ollama
import pandas as pd
from pandasai import SmartDataframe

llm = Ollama(model='mistral')

st.title('Data Analysis with PandasAI')

uploader_file = st.file_uploader("Upload a CSV file", type=["csv", 'xlsx'])

if uploader_file is not None:
    if uploader_file.name[0][-4:] == 'csv':
        data = pd.read_csv(uploader_file)
    else:
        data = pd.read_excel(uploader_file)
    st.write(data.head(5))
    df = SmartDataframe(data, config={"llm":llm})
    prompt = st.text_area("Enter your prompt:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response ..."):
                st.write(df.chat(prompt))
                
        else:
            st.warning("Please enter a prompt.")
            