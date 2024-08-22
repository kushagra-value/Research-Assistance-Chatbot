import streamlit as st
import uuid
import os
from pathlib import Path
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import traceback

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
    if 'store' not in st.session_state:
        st.session_state.store = {}

    session_options = list(st.session_state.store.keys())
    selected_session = st.selectbox("Select Session", session_options, index=0 if session_options else None)

    # Download button for the selected session
    if selected_session:
        # Ensure session history is initialized
        session_history = st.session_state.store.get(selected_session, None)
        if session_history:
            # Convert history to a text format
            session_text = "\n".join([f"{getattr(msg, 'role', 'unknown')}: {getattr(msg, 'content', 'No content')}" for msg in session_history.messages])
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

        # Generate a unique session ID if not already present
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())

        session_id = st.session_state.session_id

        # Load all PDFs from the "pdfs" folder
        pdf_files = sorted([os.path.join("pdfs", f) for f in os.listdir("pdfs") if f.endswith(".pdf")])

        if pdf_files:
            documents = []
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(pdf_file)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Error loading {pdf_file}: {str(e)}")
                    traceback.print_exc()  # Print stack trace for debugging

            if documents:
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

                # Input for user query
                user_input = st.text_input("Your question:")

                if user_input:
                    session_history = get_session_history(session_id)
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.chat_message("user").write(user_input)

                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

                        # Display the response
                        st.chat_message("assistant").write(response['answer'])
                    except Exception as e:
                        st.error(f"Error processing your request: {str(e)}")
                        traceback.print_exc()  # Print stack trace for debugging

                # Display chat history
                if session_id in st.session_state.store:
                    session_history = st.session_state.store[session_id]
                    for message in session_history.messages:
                        if message.role == "user":
                            st.chat_message("user").write(message.content)
                        elif message.role == "assistant":
                            st.chat_message("assistant").write(message.content)

        else:
            st.write("No PDFs available to process.")
    else:
        st.error("API key not set. Please check your environment configuration.")
