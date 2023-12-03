import streamlit as st
import time
from langchain.document_loaders import PyPDFLoader
from functions_back import *

document = PyPDFLoader("../documentation/documents/JavaScript.pdf")
documents = function_text_splitter(document, 500, 250)
data = pd.read_pickle("../documentation/documentation_JS.pickle")

st.title("💬 Asistente Virtual JavaScript")
st.write("""Soy un asistente virtual, que te ayuda a estudiar conceptos de JS""")  

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ingresa tu pregunta: "): #Propmt es el mensaje del usuario
    while prompt is None:
        time.sleep()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        assistant_response = function_main_app(prompt, data, documents)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})