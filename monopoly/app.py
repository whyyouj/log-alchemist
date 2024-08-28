import streamlit as st
import query_data

def process_input(question):
    ans = query_data.query_rag(question)
    return ans

st.title('Query with Ollama')
st.write('Enter a question')
question = st.text_input('Question')

if st.button('Query a Response'):
    with st.spinner('Processing...'):
        answer = process_input(question)
        st.text_area('Answer', value=answer, height=150)