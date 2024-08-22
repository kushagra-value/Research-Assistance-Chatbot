import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import ChatPromptTemplate

# New or updated import for chat history management
from langchain.chat import MessageHistory  # Replace with actual module and class

# Ensure the pdfs folder exists
import os
os.makedirs("pdfs", exist_ok=True)

# Set up embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit layout
st.set_page_config(layout="wide")
st.title("Research Assistance Chatbot")

# Retrieve Groq API Key from environment variable
api_key = os.getenv("GROQ_API_KEY")

# Custom CSS for button styling and chat messages
st.markdown("""
    <style>
    .pdf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .action-buttons {
        display: flex;
        gap: 5px;
    }
    .download-button, .delete-button {
        background-color: grey;
        border: none;
        color: white;
        cursor: pointer;
    }
    .download-button:hover, .delete-button:hover {
        background-color: red;
        color: white;
    }
    .download-button:focus, .delete-button:focus {
        outline: none;
    }
    .icon {
        width: 20px;
        height: 20px;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .chat-message {
        margin-bottom: 10px;
    }
    .user-message {
        text-align: right;
        color: #007bff;
    }
    .assistant-message {
        text-align: left;
        color: #28a745;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for available PDFs and chat history
with st.sidebar:
    st.header("Available PDFs")
    with st.expander("Manage PDFs", expanded=True):
        uploaded_files = st.file_uploader("Add a PDF", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("pdfs", uploaded_file.name)
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getvalue())
            st.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

        pdf_files = sorted(os.listdir("pdfs"))
        if pdf_files:
            for pdf in pdf_files:
                file_path = os.path.join("pdfs", pdf)
                st.markdown(
                    f"""
                    <div class="pdf-row">
                        <span>{pdf}</span>
                        <div class="action-buttons">
                            <a href="{file_path}" download="{pdf}">
                                <button class="download-button" title="Download {pdf}">
                                    <img class="icon" src="https://img.icons8.com/material-outlined/24/000000/download--v1.png"/>
                                </button>
                            </a>
                            <button class="delete-button" title="Delete {pdf}">
                                <img class="icon" src="https://img.icons8.com/material-outlined/24/000000/delete-sign.png"/>
                            </button>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.write("No PDFs available.")

    st.header("Chat History")
    session_options = list(st.session_state.store.keys()) if 'store' in st.session_state else []
    selected_session = st.selectbox("Select Session", session_options, index=0 if session_options else None)

    if selected_session:
        history = st.session_state.store.get(selected_session, None)
        if history:
            session_text = "\n".join([f"{type(msg).__name__}: {msg['content']}" for msg in history.messages])
            st.download_button(
                label="Download Session",
                data=session_text,
                file_name=f"{selected_session}_chat_history.txt",
                mime="text/plain",
            )
        else:
            st.write("No chat history available for download.")

# Main content (right column)
col1, col2 = st.columns([2, 1])

with col1:
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

        session_id = st.text_input("Session ID", value="default_session")

        if 'store' not in st.session_state:
            st.session_state.store = {}

        pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

        if pdf_files:
            documents = []
            for pdf_file in pdf_files:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Assuming `RetrievalQA` is used instead of `create_stuff_documents_chain`
            question_answer_chain = RetrievalQA(
                llm=llm,
                retriever=retriever,
                prompt=ChatPromptTemplate.from_messages(
                    [
                        ("system", "Your system prompt here."),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
            )

            chat_container = st.empty()
            chat_html = ""

            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                session_history.add_message({'content': user_input, 'type': 'user'})

                response = question_answer_chain({"input": user_input})
                response_text = response['answer']
                session_history.add_message({'content': response_text, 'type': 'assistant'})

                chat_html = ""
                for message in session_history.messages:
                    if message['type'] == 'user':
                        chat_html += f"<div class='chat-message user-message'>{message['content']}</div>"
                    else:
                        chat_html += f"<div class='chat-message assistant-message'>{message['content']}</div>"

                chat_container.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.write("No PDFs available for chat.")
    else:
        st.write("Groq API key is missing. Please set the API key.")
