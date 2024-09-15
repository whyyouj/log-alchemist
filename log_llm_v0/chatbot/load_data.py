from dotenv import load_dotenv
import os
import pymongo
from llama_index.core.ingestion import IngestionCache
from llama_index.storage.kvstore.mongodb import MongoDBKVStore as MongoDBCache
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()
MONGODB_URL = os.getenv("MONGODB_URI")
MONGODB_DBNAME = os.getenv("MONGODB_DBNAME")
MONGODB_CLIENT = pymongo.MongoClient(MONGODB_URL)
MONGODB_CACHE = IngestionCache(cache = MongoDBCache(mongo_client=MONGODB_CLIENT, db_name = MONGODB_DBNAME))
MONGODB_DOCSTORE = MongoDocumentStore.from_uri(uri=MONGODB_URL, db_name=MONGODB_DBNAME)

EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
EMBEDDINGS = HuggingFaceEmbedding(model_name = EMBED_MODEL)

# GUIDE_PDF = './data/pdf/log_Meanings_Explanation.pdf'
# LOGS_DIR = './data/log'


from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.readers.file import PDFReader

def ingest_logs(LOGS_DIR, filename):
    
    splitter = SentenceSplitter(chunk_size=180, chunk_overlap= 20)
    
    documents = SimpleDirectoryReader(LOGS_DIR, 
                                      filename_as_id=True).load_data()
    
    pipeline = IngestionPipeline(
        transformations = [splitter, EMBEDDINGS],
        vector_store = MongoDBAtlasVectorSearch(
            mongo_client = MONGODB_CLIENT,
            db_name = MONGODB_DBNAME,
            collection_name = f'{filename}_collection',
            vector_index_name = f'{filename}_idx'),
        cache = MONGODB_CACHE,
        docstore = MONGODB_DOCSTORE,
        docstore_strategy = DocstoreStrategy.UPSERTS,
        )
    
    pipeline.run(documents = documents)
    
    
def ingest_pdf(GUIDE_PDF, filename):
    
    documents = PDFReader().load_data(file=GUIDE_PDF)

    pipline = IngestionPipeline(
        transformations = [EMBEDDINGS],
        vector_store = MongoDBAtlasVectorSearch(
            mongodb_client = MONGODB_CLIENT,
            db_name = MONGODB_DBNAME,
            collection_name = f'{filename}_collection',
            vector_index_name = f'{filename}_idx'),
        cache = MONGODB_CACHE,
        docstore = MONGODB_DOCSTORE,
        docstore_strategy = DocstoreStrategy.UPSERTS,
        )
    
    pipline.run(documents=documents)
    
def test_func(i):
    print(i)
    