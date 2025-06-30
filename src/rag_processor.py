import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Use Streamlit's caching to load the model and process the PDF only once.
@st.cache_resource
def get_vector_db_from_pdf(pdf_path):
    """
    Loads a PDF, chunks it, creates embeddings, and stores it in a FAISS vector database.
    Returns the vector database object.
    """

    pdf_path = "data/Case section Tamilnadu.pdf"

    if not pdf_path:
        return None
    try:
        # 1. Load the document
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunked_docs = text_splitter.split_documents(documents)

        # 3. Create embeddings model
        # Using a powerful open-source model. It runs locally on your CPU.
        model_name = "BAAI/bge-small-en-v1.5"
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs=encode_kwargs
        )

        # 4. Create FAISS Vector Database
        # This creates an in-memory vector store.
        vector_db = FAISS.from_documents(chunked_docs, embeddings)

        print("Vector DB created successfully.")
        return vector_db

    except Exception as e:
        print(f"Error creating vector DB: {e}")
        return None