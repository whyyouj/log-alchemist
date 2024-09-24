import streamlit as st
import time
from io import StringIO
import tempfile
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]) + "/backend")
from LogSLM import LogSLM

def response_generator(res):
    for word in res.split():
        yield word + " "
        time.sleep(0.05)

st.title("Log Alchemist")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_paths" not in st.session_state:
    st.session_state.file_paths = []

if "logslm" not in st.session_state:
    st.session_state.logslm = LogSLM()

with st.container(height=200):
    uploaded_files = st.file_uploader("Upload your log files", type=['pdf', 'txt', 'log', 'docx', 'csv'], 
                                      accept_multiple_files=True, label_visibility='visible')
    st.session_state.file_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name
            st.session_state.file_paths.append(file_path)
            print('FILE PATHS: ', st.session_state.file_paths)

    if len(st.session_state.file_paths) > 0:
        st.session_state.logslm.ingest(st.session_state.file_paths)    
        
    for uploaded_file in uploaded_files:    
        os.remove(file_path)

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# accept user input
if prompt := st.chat_input("Ask me anything about your logs!"):
    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # display slm response in chat message container
    with st.chat_message("assistant"):
        slm_res = st.session_state.logslm.ask(prompt)
        response = st.write_stream(response_generator(slm_res))

    # add slm response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})