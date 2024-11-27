import streamlit as st
import os
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

# Title of the app
st.title("Chat with Groq!")
st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. Let's start our conversation!")

# CSS for styling
st.markdown(
    """
    <style>
    .chat-bubble-user {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 60%;
        margin-left: auto;
        word-wrap: break-word;
    }
    .chat-bubble-ai {
        background-color: #E4E6EB;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 60%;
        margin-right: auto;
        word-wrap: break-word;
    }
    .chat-container {
        height: 400px;
        overflow-y: scroll;
        padding-right: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #F9F9F9;
        padding-left: 10px;
    }
    .user-input {
        width: 100%;
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 10px;
        margin-top: 10px;
    }
    .new-chat-button {
        width: 100%;
        padding: 12px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
    }
    .new-chat-button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for chat history and memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True
    )

# Create a chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display the chat history
for message in st.session_state.chat_history:
    if message["AI"]:
        st.markdown(f'<div class="chat-bubble-ai">{message["AI"]}</div>', unsafe_allow_html=True)
    if message["human"]:
        st.markdown(f'<div class="chat-bubble-user">{message["human"]}</div>', unsafe_allow_html=True)

# Close the chat container
st.markdown('</div>', unsafe_allow_html=True)

# Text input field for user question
user_question = st.text_input("Ask a question:", key="user_input", placeholder="Type your message here...")

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.1-8b-instant"
)

if user_question:
    # Save previous interactions to memory
    memory = st.session_state.memory
    for message in st.session_state.chat_history:
        memory.save_context(
            {"input": message["human"]},
            {"output": message["AI"]}
        )
    
    prompt = ChatPromptTemplate.from_messages(
        [ 
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
            HumanMessagePromptTemplate.from_template("{human_input}"),  # Template for user input
        ]
    )

    conversation = LLMChain(
        llm=groq_chat,  # Groq LangChain chat object
        prompt=prompt,  # Constructed prompt template
        verbose=True,   # Enables verbose output
        memory=memory,  # Conversational memory object
    )

    response = conversation.predict(human_input=user_question)
    message = {"human": user_question, "AI": response}
    st.session_state.chat_history.append(message)

    # Display the new chatbot response
    st.markdown(f'<div class="chat-bubble-ai">{response}</div>', unsafe_allow_html=True)

# New chat button
if st.button("New Chat", key="new_chat", help="Reset the chat history and start a new conversation"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    st.write("Session reset successfully!")
