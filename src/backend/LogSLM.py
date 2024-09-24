from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_unstructured import UnstructuredLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

class LogSLM:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.model = ChatOllama(model="llama3.1")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.rag_prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            <context>
            {context}
            </context>

            Answer the following question:

            {question}
            """
        )
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ingest(self, file_paths):
        # NOTE: unable to use both UnstructuredLoader and DirectoryLoader as they require the python-magic/python-libmagic/python-magic-bin
        # packages which I have tried installing but still faced issues using the loaders
        
        # loader = UnstructuredLoader(file_paths)
        # directory = '/'.join(file_paths[0].split('/')[:-1])
        # print('DIRECTORY: ', directory)                      
        # loader = DirectoryLoader(f"{directory}/")

        loader = TextLoader(file_paths[0])
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=local_embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.1,
            },
        )

        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        if not self.rag_chain:
            return self.model.invoke(query) #"Please, add a PDF document first."

        return self.rag_chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None