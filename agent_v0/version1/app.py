import streamlit as st
from graph.lang_graph import Graph

st.title("Chat 🦾")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

model = Graph().get_graph()    
if prompt := st.chat_input("Ask a Question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.spinner('Processing'):
        # if llm == ':rainbow[llama3]':
        #     model_name = 'llama3'
        # elif llm == '***llama3-groq-tool-use***':
        #     model_name = 'llama3-groq-tool-use'
        # elif llm == 'mistral':
        #     model_name = 'mistral'
        # else:
        #     model_name = 'phi'
        
        # model = Graph().get_graph()
        response = model.invoke({"input":prompt})["agent_out"]
        # response_str = ''
        # for r in response.response_gen:
        #     response_str += r
        response_str = response
        #st.text_area('Resposne:', value = response_str, height = 150)
    response = f"{'Bot'}: {response_str}" 
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content":response})