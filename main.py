import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# Initialize the LLM
groq_api_key = os.getenv("GROQ_API_KEY")  # Replace with your actual API key or environment variable
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile"  # Adjust model name as required
)

# Initialize memory to store the conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the conversation prompt
prompt_template = """
Human: {input}
AI: {chat_history}
"""
prompt = PromptTemplate(input_variables=["input", "chat_history"], template=prompt_template)

# Initialize the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Function to process input from a file (either PDF or text file)
def process_file(file):
    try:
        # For PDF files, we'll extract text content here (you can use PyPDF2 or pdfplumber for PDF text extraction)
        if file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        # For text files, simply read the content
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        else:
            return "Unsupported file type"
    except Exception as e:
        return f"Error reading the file: {str(e)}"

# Streamlit UI
def chatbot_ui():
    st.set_page_config(page_title="AI Chatbot", layout="wide")

    # Sidebar for instructions
    st.sidebar.title("AI Chatbot")
    st.sidebar.write("Upload a file (PDF or text) and ask questions!")

    # Main UI
    st.title("AI Chatbot")
    st.write("Upload a file and ask questions based on its content!")
    st.markdown("---")

    # Chat history state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "file_content" not in st.session_state:
        st.session_state.file_content = []

    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(chat)

    st.markdown("---")

    # File upload button and text input box at the bottom
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("###")
        uploaded_file = st.file_uploader("Upload a PDF or text file:", type=["pdf", "txt"], label_visibility="collapsed")
    with col2:
        question = st.text_input("Ask a question:", key="user_input", label_visibility="collapsed")

    # Button to process file
    if uploaded_file:
        file_content = process_file(uploaded_file)
        st.session_state.file_content = file_content
        st.session_state.chat_history.append("**File uploaded successfully!**")
        st.success("File content successfully uploaded. You can now ask questions based on the content.")

    # Process question
    if question:
        context = st.session_state.file_content if st.session_state.file_content else ""
        full_input = f"{context}\n\n{question}" if context else question

        # Generate the model's response
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        response = llm_chain.run(input=full_input, chat_history=chat_history)

        # Extract only the content of the latest AI response
        ai_response = response.split("AI: ")[-1].strip()

        # Update chat history
        st.session_state.chat_history.append(f"**You:** {question}")
        st.session_state.chat_history.append(f"**AI:** {ai_response}")

        # Clear the input box
        st.session_state.user_input = ""  # Reset the input field without rerunning

if __name__ == "__main__":
    chatbot_ui()
