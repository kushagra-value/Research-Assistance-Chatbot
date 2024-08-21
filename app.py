import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Ensure the pdfs folder exists
os.makedirs("pdfs", exist_ok=True)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit layout
st.set_page_config(layout="wide")
st.title("Conversational RAG With PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content")

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
        background-color: transparent;
        border: none;
        color: #007bff;
        cursor: pointer;
    }
    .download-button:hover, .delete-button:hover {
        color: #0056b3;
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

# Main content and sidebar
col1, col2 = st.columns([2, 1])

# Main content (left column)
with col1:
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

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

            # System prompt for contextualizing the question
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            # System prompt for answering the question
            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.text_input("Your question:")
            if user_input:
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    },
                )
                st.write("Assistant:", response['answer'])
                st.write("Chat History:", session_history.messages)
        else:
            st.warning("No PDFs available in the 'pdfs' folder.")
    else:
        st.error("Groq API Key not found in the environment. Please set it in your environment variables.")

# Sidebar (right column)
with col2:
    st.header("Available PDFs")

    # Upload PDFs in the right column
    uploaded_files = st.file_uploader("Add a PDF", type="pdf", accept_multiple_files=True)

    # Process and save uploaded PDFs to the "pdfs" folder
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join("pdfs", uploaded_file.name)
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
        st.success(f"Uploaded {len(uploaded_files)} file(s) to the 'pdfs' folder.")
