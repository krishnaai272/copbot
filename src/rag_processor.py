import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- THIS IS THE CORRECT FUNCTION NAME THAT app.py IS LOOKING FOR ---
@st.cache_resource
def create_vector_db_from_pdfs(pdf_filepaths: list[str]):
    """
    Loads PDFs, chunks them, and creates a FAISS vector store.
    This is much faster to initialize than the ParentDocumentRetriever.
    """
    all_documents = []
    for filepath in pdf_filepaths:
        if not os.path.exists(filepath):
            st.warning(f"File not found: {filepath}. Skipping.")
            continue
        try:
            loader = PyPDFLoader(file_path=filepath)
            all_documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {filepath}: {e}")
            continue

    if not all_documents:
        st.error("No documents were loaded.")
        return None

    try:
        # Use a single, effective chunking strategy.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunked_docs = text_splitter.split_documents(all_documents)

        embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        st.write(f"Creating vector database from {len(chunked_docs)} chunks...")
        
        # Create the FAISS vector store from the chunks.
        vector_db = FAISS.from_documents(chunked_docs, embeddings_model)
        
        st.success("Knowledge base initialized successfully!")
        return vector_db

    except Exception as e:
        st.error(f"An error occurred while creating the knowledge base: {e}")
        return None