# %pip install --upgrade --quiet  langchain langchain-community  langchain-experimental neo4j tiktoken yfiles_jupyter_graphs 
from langchain_groq import ChatGroq
from langchain.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings



# Get the llm
def get_llm():
    # initializing the model
    llm = ChatGroq(model="Llama-3.1-70b-Versatile", temperature=0)
    return llm

# Get the embedding model
def get_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# Loading target data

## Loading data from wikipedia
def wiki_loader(query:str):
    raw_documents = WikipediaLoader(query=query).load()

    return raw_documents

## Loading data from pdfs:
def pdf_loader(path:str):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

# Spliting data into chunks
def data_spliter(docs,chunk_size=512,chunk_overlap=24):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# Loading data into a vector database (FAISS)
def load_into_vector_db(documents,embeddings):   
    vectorstore = FAISS.from_documents(documents=documents,embedding= embeddings)
    # Save the data localy in the following folder
    store_dir = "data_store"
    vectorstore.save_local(store_dir)   
    return vectorstore


