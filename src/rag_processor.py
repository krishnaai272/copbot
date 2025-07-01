import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
INDEX_LOAD_PATH = "faiss_index"

@st.cache_resource
def load_vector_db():
    """
    Loads the pre-built FAISS index from disk. This is very fast.
    """
    if not os.path.exists(INDEX_LOAD_PATH):
        st.error(f"The FAISS index folder was not found at '{INDEX_LOAD_PATH}'. Please run 'build_index.py' first.")
        return None
    
    try:
        st.write("Loading pre-built knowledge base...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load the vector store from the local folder
        vector_db = FAISS.load_local(
            INDEX_LOAD_PATH, 
            embeddings_model, 
            allow_dangerous_deserialization=True # This is needed to load FAISS indexes
        )
        
        st.success("Knowledge base loaded successfully!")
        return vector_db

    except Exception as e:
        st.error(f"An error occurred while loading the knowledge base: {e}")
        return None