from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def create_rag_chain_from_retriever(retriever):
    """
    Creates a RAG chain directly from a pre-configured retriever object.
    """
    # 1. Initialize the LLM
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # 2. Create the RAG Chain
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
    
    # The final chain uses the retriever that was passed directly into this function
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain