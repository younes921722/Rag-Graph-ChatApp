from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import time
from core import full_chain,pdf_loader
from loader import load_docs_into_vector_db, load_docs_into_graph_db

# Function to simulate streaming a string
def stream_string(text, delay=0.02):
    placeholder = st.empty()
    stream_text = ""
    for char in text:
        stream_text += char
        placeholder.text(stream_text)
        time.sleep(delay)

# Saving chat history in streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content = "Hello! I am an assistant based on kownledge graph to assist you know more about the prophet Muhammad (PBUH)?"
                  +", do you have any question?"),
    ]

st.set_page_config(page_title="Knowledge graph", page_icon=":book:" )
st.title("Ask about Muhammad (PBUH)")


# Setting the sidebar
with st.sidebar:
    uploaded_files = st.file_uploader("Upload your files", type=["pdf"])
    print("9999999999999999999",uploaded_files)
    if st.button("Load To Vecotr-DB"):
        load_docs_into_vector_db(uploaded_files=uploaded_files)
        

    if st.button("Load into Graph-DB"):
        load_docs_into_graph_db(uploaded_files=uploaded_files)
        st.success("Data saved into the Graph Db successfully :)")

# Setting the Main chat page
## Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content )

## React to user input
user_input=st.chat_input("start asking!")
if user_input is not None and user_input.strip() !="":
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)
    
    with st.chat_message("Ai"):
        st.write(st.session_state.chat_history)
        response = full_chain(user_input, chat_history=st.session_state.chat_history)
        if response is not None and response!="":
            st.session_state.chat_history.append(AIMessage(content=response))
            st.markdown(response)
