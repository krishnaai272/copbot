from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def create_rag_chain(retriever):
    """
    Creates a Retrieval-Augmented Generation (RAG) chain.
    This chain will only use the provided document retriever for context.
    """

    # 1. Initialize the LLM (Groq in this case)
    # Using a powerful model like Llama3 70b is great for reasoning
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    # 2. Create a specific prompt template
    # This prompt strictly instructs the LLM to use ONLY the provided context.
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

    # 3. Create the main document chain
    # This chain knows how to combine the retrieved documents into the prompt.
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 4. Create the final retrieval chain
    # This orchestrates the whole process: retrieves docs, then passes them to the document_chain.
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain