import os
# --- CHANGE: Import the more robust loader ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
PDF_FOLDER_PATH = "data"
INDEX_SAVE_PATH = "faiss_index"

def build_and_save_index():
    print("Starting the index building process...")

    pdf_files = [os.path.join(PDF_FOLDER_PATH, f) for f in os.listdir(PDF_FOLDER_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the 'data' folder. Exiting.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    
    all_documents = []
    for filepath in pdf_files:
        try:
            # --- CHANGE: Use PyMuPDFLoader instead of PyPDFLoader ---
            loader = PyMuPDFLoader(file_path=filepath)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    if not all_documents:
        print("No documents could be loaded from the PDF files. Exiting.")
        return

    print(f"Loaded a total of {len(all_documents)} pages.")

    try:
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunked_docs = text_splitter.split_documents(all_documents)
        print(f"Created {len(chunked_docs)} text chunks.")

        print("Initializing embedding model (BAAI/bge-small-en-v1.5)...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("Creating FAISS vector store from chunks... (This will take several minutes)")
        vector_db = FAISS.from_documents(chunked_docs, embeddings_model)
        
        print(f"Vector store created. Saving index to '{INDEX_SAVE_PATH}'...")
        vector_db.save_local(INDEX_SAVE_PATH)
        
        print("\nSUCCESS! The FAISS index has been built and saved locally.")

    except Exception as e:
        print(f"\nAn error occurred during the process: {e}")

if __name__ == "__main__":
    build_and_save_index()