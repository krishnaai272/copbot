import os
import streamlit as st
import faiss  # <--- FIX 1: IMPORT THE FAISS LIBRARY
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def create_knowledge_base_from_pdfs(pdf_filepaths: list[str]):
    """
    Loads and processes a list of PDF files, creates a unified vector database
    using the ParentDocumentRetriever strategy for more accurate context.
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
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

        embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # --- START: THIS IS THE FIX ---
        # 2. Define the embedding dimension and create an empty FAISS index.
        embedding_dimension = 384  # Dimension for BAAI/bge-small-en-v1.5
        faiss_index = faiss.IndexFlatL2(embedding_dimension)
        
        # 3. Create a compliant document store for FAISS to use.
        faiss_docstore = InMemoryDocstore({})
        
        # 4. Create the vector store, passing it the pre-built empty index.
        vectorstore = FAISS(
            embedding_function=embeddings_model,
            index=faiss_index,          # Pass the created empty index here
            docstore=faiss_docstore,
            index_to_docstore_id={}
        )
        # --- END: THIS IS THE FIX ---

        # The storage layer for the large parent documents
        store = InMemoryStore()

        st.write(f"Initializing Parent Document Retriever from {len(all_documents)} pages...")
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        retriever.add_documents(all_documents, ids=None)
        st.success("Knowledge base initialized successfully using Parent Document Retriever!")
        return retriever

    except Exception as e:
        st.error(f"An error occurred while creating the knowledge base: {e}")
        return None