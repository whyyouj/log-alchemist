import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = 'chroma'
def process_input(urls, question):
    model_local = Ollama(model='mistral')
    
    url_list = urls.split('\n')
    docs = [WebBaseLoader(url).load() for url in url_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex=False,
    )
    doc_split = text_splitter.split_documents(docs_list)
    
    db = Chroma.from_documents(
        documents = doc_split,
        collection_name = 'db_a',
        embedding= OllamaEmbeddings(model = 'nomic-embed-text')
    )
    retriever = db.as_retriever()
    
    after_rag_template = """answer the question based only on the following text"""
    
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

st.title('document query with Ollama')
st.write('enter urls and a question to query the documents')

Urls = st.text_area('Enter URLs seperated by new lines', height= 150)
question = st.text_input("Question")

if st.button('query documents'):
    with st.spinner('Processing...'):
        answer = process_input(Urls, question)
        st.text_area('Answer', value=answer, height = 150, disable = True)        
    