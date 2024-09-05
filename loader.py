from core import get_embedding, pdf_loader, data_spliter, load_into_vector_db, convert_to_graph, get_llm, store_graph, documents_spliter

# Load only uploaded pdf files from stremlit into Vector-DB
def load_docs_into_vector_db(uploaded_files):
    embeddings = get_embedding
    docs = pdf_loader(uploaded_files)
    splited_docs = data_spliter(docs=docs)
    load_into_vector_db(texts=splited_docs,embeddings=embeddings)

# Load files into Graph-DB
def load_docs_into_graph_db(uploaded_files):
    # get LLM
    llm = get_llm()
    # Convert docs to graph
    docs = pdf_loader(uploaded_files)
    splited_docs = documents_spliter(docs=docs[:2])
    ## Creating and Saving the graph data in our graph db
    store_graph(llm=llm, documents=splited_docs)
    print("done from Graph Db")

