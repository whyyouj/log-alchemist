from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.ollama import OllamaEmbeddings
import pandas as pd
import argparse
# llm= Ollama(model='mistral')
# agent_executer = create_csv_agent(llm, 'data/PharmaDrugSales.csv', verbose = True)
# agent_executer.invoke("How many Day are there")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    print(query_rag(query_text))

def query_rag(question):
    embedding_function = OllamaEmbeddings(model = 'nomic-embed-text')
    loader = CSVLoader('data/PharmaDrugSales.csv')
    data = loader.load()
    db = Chroma.from_documents(
                documents=data,
                collection_name='db_a',
                embedding=embedding_function
            )
    results = db.similarity_search_with_score(question, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    answer = Ollama(model='mistral').invoke(prompt)
    return answer

def get():
    df = pd.read_csv('data/PharmaDrugSales.csv')
    a = df['SalicylicAcidDerivatives']
    print(a)
    print(a)
    return
if __name__ == '__main__':
    get()