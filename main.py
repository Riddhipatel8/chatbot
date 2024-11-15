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
    
st.title("Chat with Groq!")
st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. Let's start our conversation!")

    # Initialize session state for chat history and memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True
        )
    
user_question = st.text_input("Ask a question:")
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
st.write("Chatbot:", response)

for message in st.session_state.chat_history:
        st.write(f"**You:** {message['human']}")
        st.write(f"**Chatbot:** {message['AI']}")
        
if st.button("Reset"):
    st.session_state.chat_history = []
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True
     )
st.write("Session reset successfully!")
