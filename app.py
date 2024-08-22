from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
import streamlit as st

# Ensure the pdfs folder exists
os.makedirs("pdfs", exist_ok=True)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit layout
st.set_page_config(layout="wide")
st.title("Research Assistance Chatbot")

# Retrieve Groq API Key from environment variable
api_key = os.getenv("GROQ_API_KEY")

# Custom CSS for button styling
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
    </style>
""", unsafe_allow_html=True)

# Sidebar for available PDFs and chat history
with st.sidebar:
    st.header("Available PDFs")
    with st.expander("Manage PDFs", expanded=True):
        # Upload PDFs in the expandable section
        uploaded_files = st.file_uploader("Add a PDF", type="pdf", accept_multiple_files=True)

        # Process and save uploaded PDFs to the "pdfs" folder
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join("pdfs", uploaded_file.name)
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getvalue())
            st.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")

        # Display and download PDFs alphabetically
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
    # Dropdown for selecting session
    session_options = list(st.session_state.store.keys()) if 'store' in st.session_state else []
    selected_session = st.selectbox("Select Session", session_options, index=0 if session_options else None)

    # Download button for the selected session
    if selected_session:
        history = st.session_state.store.get(selected_session, None)
        if history:
            # Convert history to a text format
            session_text = "\n".join([f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content')}" for msg in history.messages])
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
        llm = HuggingFaceLLM(model_name="Gemma2-9b-It")

        # Chat interface
        session_id = st.text_input("Session ID", value="default_session")

        # Statefully manage chat history
        if 'store' not in st.session_state:
            st.session_state.store = {}

        # Load all PDFs from the "pdfs" folder
        pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

        if pdf_files:
            documents = []
            for pdf_file in pdf_files:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)

            # Split and create embeddings for the documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # Set up the prompt for QA
            prompt_template = PromptTemplate(
                template="You are an assistant for question-answering tasks. Use the following context to answer the question: {context} Answer: {input}",
                input_variables=["context", "input"]
            )

            question_answer_chain = RetrievalQA(
                llm=llm,
                retriever=retriever,
                combine_docs_chain=prompt_template
            )

            # Manage history and responses
            def conversational_rag_chain(input_text: str, session_id: str):
                session_history = get_session_history(session_id)
                # Simulate retrieval and QA chain
                response = question_answer_chain({"context": "Context from retriever", "input": input_text})
                session_history.add_user_message(input_text)
                session_history.add_ai_message(response)
                return response

            def get_session_history(session: str):
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            # Display the full chat history
            st.markdown("### Conversation History")
            session_history = get_session_history(session_id)

            for message in session_history.messages:
                role = 'User' if message.get('role') == 'user' else 'Assistant'
                st.markdown(f"**{role}:** {message.get('content')}")

            user_input = st.text_area("Your question:", key="user_input")

            if st.button("Send"):
                if user_input:
                    response = conversational_rag_chain(user_input, session_id)
                    st.session_state.user_input = ""  # Clear input field after submission
                    st.experimental_rerun()  # Refresh to show new messages in the chat history
        else:
            st.warning("No PDFs available in the 'pdfs' folder.")
    else:
        st.error("Groq API Key not found in the environment. Please set it in your environment variables.")

with col2:
    st.write("Sidebar content")
