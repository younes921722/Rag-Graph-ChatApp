# %pip install --upgrade --quiet  langchain langchain-community  langchain-experimental neo4j tiktoken yfiles_jupyter_graphs 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import os
from neo4j import GraphDatabase
from yfiles_jupyter_graphs import GraphWidget
from entities import Entities
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_core.documents import Document


# loading envirement variables
load_dotenv()


# Get the llm
def get_llm():
    # initializing the model
    llm = ChatGroq(model="Llama-3.1-70b-Versatile", temperature=0)
    return llm
llm = get_llm()

# Get the embedding model
def get_embedding():
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_name="sentence-transformers/all-mpnet-base-v2"
    # model_name = "all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

emb = get_embedding()
# Get the Neo4j graph instance
graph = Neo4jGraph()

# Loading target data

## Loading data from wikipedia
def wiki_loader(query:str):
    raw_documents = WikipediaLoader(query=query).load()

    return raw_documents

## Loading data from pdfs:
def pdf_loader(pdf_docs):
    # get pdf text
    if pdf_docs is not None:
        pdf_reader = PdfReader(pdf_docs)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Spliting text data into chunks
def data_spliter(docs,chunk_size=512,chunk_overlap=24):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(docs)

# Spliting Documents data inot chunks
def documents_spliter(docs,chunk_size=512,chunk_overlap=24):
    # Convert text to a Document object with an attribute page_content to be plitted by split_documents..
    docs = [Document(page_content=docs)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# Loading data into a vector database (FAISS)
def load_into_vector_db(texts,embeddings):   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=texts,embedding=embeddings)
    # Save the data localy in the following folder
    store_dir = "data_store"
    vectorstore.save_local(store_dir)

# Loading data from existing vectoreStore
def vector_db_retriever(embeddings,path="data_store"):
    return FAISS.load_local(folder_path=path,embeddings=embeddings,allow_dangerous_deserialization=True).as_retriever()

# Transforming our docs into a graph data
## Convert docs to graph    
def convert_to_graph(documents, llm=llm):
    llm_transformer = LLMGraphTransformer(llm=llm)
    return llm_transformer.convert_to_graph_documents(documents)

## Saving graph data in our graph db
def store_graph(llm, documents):
    graph_documents = convert_to_graph(llm, documents)
    graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
                )
# Show the graph
def showGraph():
    driver = GraphDatabase.driver(
        uri = os.environ["NEO4J_URI"],
        auth = (os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"]))
    session = driver.session()
    widget = GraphWidget(graph = session.run("MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t").graph())
    widget.node_label_mapping = 'id'
    return widget

# Entity chain
def entity_chain(question:str):    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)
    return entity_chain.invoke({"question":question})

# Generate query
def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query)
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output
            """,
            {"query": generate_full_text_query(entity)}
        )
        result += "\n".join([el['output'] for el in response])
    return result

# Generating the full context (include the cypher query result and the data returned from the vector db )
def full_retriever(question: str):
    embeddings= get_embedding()
    vectorRetriever = vector_db_retriever(embeddings=embeddings)
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vectorRetriever.invoke(question)]
    print("*"*10,vector_data)
    print("vector data: ","*"*8,vector_data)
    final_data = f"""Graph data:
                    {graph_data}
                    vector data:
                    {"#Document ". join(vector_data)}
                        """
    return final_data

# Full Chain
def full_chain(user_input:str, chat_history:list):
    chat_history = f"{chat_history}"
    template = """Answer the question based only on the following context:
                {context}
                Make sure that all the answers must be according to the Sunni view.
                Be kind, clear and give detailed information.
                Chat history: {chat_history}
                Question: {question}
                Use natural language and be concise.
                Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Convert chat_history to a string, handling both AIMessage and HumanMessage instances
    # chat_history_str = "\n".join([str(item) for item in chat_history]) if isinstance(chat_history, list) else str(chat_history)

    chain = (
            {
                "context": full_retriever,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
        | prompt
        | llm
        | StrOutputParser()
            )
    # # response = chain.invoke(input=user_input)
    # response = chain.invoke(input={"question":user_input,"chat_history":chat_history})
    # return response
     # Debugging prints
    print("User Input:", user_input)
    print("Chat History:", chat_history)
    
    # Ensure the input structure is correct
    response = chain.invoke({"question":user_input,"chat_history":chat_history})
    
    print("Response:", response)
    return response
