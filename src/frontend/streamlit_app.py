import streamlit as st
import requests
import json
import time
from io import StringIO

api_base_url = "https://2a3a-104-155-200-8.ngrok-free.app"#"http://127.0.0.1:8000"

def response_generator(res):
    for word in res.split():
        yield word + " "
        time.sleep(0.05)

st.title("Log Alchemist")

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.container(height=200):
    uploaded_files = st.file_uploader("Upload your log files", type=['pdf', 'txt', 'log', 'docx', 'csv'], 
                                      accept_multiple_files=True, label_visibility='visible')
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        print("filename:", uploaded_file.name)
        print('bytes: ', bytes_data)

        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        print('stringio: ', stringio)
        string_data = stringio.read()
        print('string data: ', string_data)

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
        json_params = {"prmpt": prompt, "messages": st.session_state.messages}
        slm_res = requests.post(f'{api_base_url}/run_slm', json=json_params, headers = {"Content-Type":"application/json"}).json()
        response = st.write_stream(response_generator(slm_res['res']))
        # response = st.write_stream(response_generator(f"Back at you: {prompt}"))

    # add slm response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})