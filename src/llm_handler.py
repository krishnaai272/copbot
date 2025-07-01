from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def create_effective_rag_chain(vector_db):
    """
    Creates a highly effective RAG chain using a fast vector search
    followed by a powerful local reranker.
    """
    # 1. Initialize the LLM for the final answer
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # 2. Create the Retriever with Reranking
    
    # 2a. Base Retriever: A simple vector store retriever that fetches many results.
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 15})

    # 2b. Reranker: Initialize the free, local Cross-Encoder model.
    reranker_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    compressor = CrossEncoderReranker(model=reranker_model, top_n=4) # Keep the top 4 best results

    # 2c. Compression Retriever: The final, effective retriever.
    # It gets 15 results from the base_retriever, then uses the reranker to pick the best 4.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    # 3. Create the RAG Chain
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert assistant for the Thoothukudi District Police.
        Answer the user's question based ONLY on the following context.
        If the information is not in the context, respond with:
        "The answer is not available in the provided documents. Please try another question."
        Do not use any outside knowledge. Be concise and helpful.

        <context>
        {context}
        </context>

        Question: {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(compression_retriever, document_chain)

    return retrieval_chain