# Knowledge Graph Assistant for Muhammad (PBUH)

This project is a **Streamlit-based application** that uses a **knowledge graph** to answer questions about the Prophet Muhammad (PBUH). The app allows users to upload PDF documents and store the data in either a vector database or a graph database. The assistant responds to user queries based on the knowledge from these documents using AI-powered language models.

In this implementation, the project is focused on files related to the Prophet Muhammad (PBUH), but users can upload **their own data** to the graph database (Neo4j) and work with it to perform custom queries and AI-based interactions.

## Features

- **AI-Powered Chat**: The assistant uses AI to provide insightful responses.
- **PDF Upload**: Users can upload PDF files, which are processed and stored in the backend databases.
- **Vector Database**: Store and query documents in a vector-based database.
- **Graph Database (Neo4j)**: Load custom documents into a graph database for enhanced knowledge retrieval.
- **Stream Responses**: The app simulates real-time streaming of chat responses for a better user experience.

## Technologies Used

- **Streamlit**: A Python-based framework for building data-driven web apps.
- **LangChain**: Used for handling AI-based chat responses and interaction with the knowledge graph.
- **Vector Database**: To store documents for fast retrieval based on embeddings.
- **Graph Database (Neo4j)**: For storing relationships between entities in the documents.
- **Python**: The core language used to build the app.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/younes921722/Rag-Graph-ChatApp
   ```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
Copier le code
streamlit run app.py
```
## Usage

- Upload a PDF file through the sidebar.
- Choose to either load the file into the Vector Database or the Graph Database (Neo4j).
- Ask questions about Prophet Muhammad (PBUH) or your own data in the main chat window, and the AI assistant will respond based on the uploaded documents.
## How It Works
- **Document Loading**: Upload PDFs to populate either the vector or graph database.
- **Custom Data**: Users can upload their own data to Neo4j, enabling them to perform queries and interactions based on their custom dataset.
- **Chat Interaction**: The assistant answers questions based on the loaded documents using natural language processing (NLP) models.
- **Real-time Updates**: Responses are streamed in real-time for a more interactive user experience.

## Contribution
Feel free to fork the repository and contribute to the project by creating pull requests.

## License
This project is licensed under the Apache License.
