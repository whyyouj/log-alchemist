import streamlit as st
from load_data import ingest_logs, ingest_pdf, test_func
from pathlib import Path
import os 
import time
from small_llm import get_chat_engine

st.title("Log Analysis 📊")

col1, col2 = st.columns([5, 0.5])

with col1:
    st.markdown("**Upload file:**")
    with st.form(key = 'Form :', clear_on_submit=True):
        Name = st.text_input("File Name: ")
        File = st.file_uploader(label = "Upload file", type = ['pdf', 'logs'])
        Submit = st.form_submit_button(label = "Submit")


    if Submit:
        if Name == '':
            warning = st.warning('**Please file in the File Name**')
            time.sleep(3)
            warning.empty()
        
        else:
            save_folder = f"../data/{Name}"
            
            os.makedirs(save_folder, exist_ok= True)
                
            save_path = Path(save_folder,File.name)

            with open(save_path, mode = 'wb') as w:
                w.write(File.getvalue())
            
            if save_path.exists():
                success_message = st.success(f"File {File.name} uploaded successfully")
                time.sleep(3)
                success_message.empty()
                if File.name.endswith('.pdf'):
                    ingest_pdf(save_path, Name)
                    success_message = st.success(f"File upload in MongoDB with collection name: {Name}")
                    time.sleep(3)
                    success_message.empty()
                elif File.name.endswith('.log'):
                    ingest_logs(save_folder, Name)
                    success_message = st.success(f"File upload in MongoDB with collection name: {Name}")
                    time.sleep(3)
                    success_message.empty()
                else:
                    warning = st.warning('**Please Upload PDF/Logs file only**')
                    time.sleep(3)
                    warning.empty()
      
st.markdown("**Chat with a Small LLM**:")
llm = st.radio(label="Choose your llm:", 
         options = [':rainbow[llama3]', '***llama3-groq-tool-use***', 'mistral', 'phi'])


st.title("Chat 🦾")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Ask a Question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.spinner('Processing'):
        if llm == ':rainbow[llama3]':
            model_name = 'llama3'
        elif llm == '***llama3-groq-tool-use***':
            model_name = 'llama3-groq-tool-use'
        elif llm == 'mistral':
            model_name = 'mistral'
        else:
            model_name = 'phi'
        
        model = get_chat_engine(model_name)
        response = model.stream_chat(prompt)
        response_str = ''
        for r in response.response_gen:
            response_str += r
        #st.text_area('Resposne:', value = response_str, height = 150)
    response = f"{model_name}: {response_str}" 
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content":response})