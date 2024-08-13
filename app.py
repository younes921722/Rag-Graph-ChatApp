from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import time

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
        try:
            db = st.session_state.db
            sql_chain = get_sql_chain(db)
            # response = get_response(user_query=user_input, db=st.session_state.db, chat_history=st.session_state.chat_history)
            response = st.write_stream(get_response(user_query=user_input, db=st.session_state.db, chat_history=st.session_state.chat_history))
            if response is not None and response!="":
                st.session_state.chat_history.append(AIMessage(content=response))
        except:
            stream_string("please try to upload your csv files first!")
