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

# Load environment variables from the .env file
load_dotenv()

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')

    # The title and greeting message of the Streamlit application
    st.title("Chat with Groq!")
    st.write("Hello! I'm your friendly Groq chatbot. I can help answer your questions, provide information, or just chat. I'm also super fast! Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:", "You are a helpful assistant.")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question:")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # If the user has asked a question,
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),  # Persistent system prompt
                MessagesPlaceholder(variable_name="chat_history"),  # Placeholder for chat history
                HumanMessagePromptTemplate.from_template("{human_input}"),  # Template for user input
            ]
        )

        # Create a conversation chain using the LangChain LLM
        conversation = LLMChain(
            llm=groq_chat,  # Groq LangChain chat object
            prompt=prompt,  # Constructed prompt template
            verbose=True,   # Enables verbose output
            memory=memory,  # Conversational memory object
        )
        
        # Generate the chatbot's response
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(f"**You:** {message['human']}")
        st.write(f"**Chatbot:** {message['AI']}")

    # Reset chat history
    if st.button("Reset"):
        st.session_state.chat_history = []
        st.write("Session reset successfully!")

if __name__ == "__main__":
    main()

