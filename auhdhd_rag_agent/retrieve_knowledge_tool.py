import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone as PineconeClient # Not strictly needed if PineconeVectorStore handles client init

# Load environment variables from .env file at module level
load_dotenv()

# Define constants (optional, but good practice)
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K_RESULTS = 3

def query_knowledge_base(user_query: str) -> dict:
    """
    Retrieves relevant information from the knowledge base hosted on Pinecone to answer the user's query.
    Uses OpenAI to generate embeddings for the query before searching.
    Use this tool when you need to find specific information or context to respond to the user.
    Args:
        user_query (str): The user's query or question to search for in the knowledge base.
    Returns:
        dict: A dictionary containing the status and the retrieved context.
              Example: {"status": "success", "retrieved_context": "Relevant text..."}
                       or {"status": "error", "message": "Error details..."}
    """
    print(f"Received query for Pinecone: {user_query}")

    # Retrieve API keys and Pinecone configuration from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # PINECONE_ENVIRONMENT is used by Pinecone client implicitly if set as an env var.
    # Not directly passed to PineconeVectorStore.from_existing_index in latest versions.
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT") 
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    # Validate environment variables
    if not all([openai_api_key, pinecone_api_key, pinecone_environment, pinecone_index_name]):
        missing_vars = []
        if not openai_api_key: missing_vars.append("OPENAI_API_KEY")
        if not pinecone_api_key: missing_vars.append("PINECONE_API_KEY")
        if not pinecone_environment: missing_vars.append("PINECONE_ENVIRONMENT")
        if not pinecone_index_name: missing_vars.append("PINECONE_INDEX_NAME")
        error_message = f"Error: Missing required environment variables: {', '.join(missing_vars)}"
        print(error_message)
        return {"status": "error", "message": error_message, "retrieved_context": ""}

    print("Initializing OpenAI embeddings model...")
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model=OPENAI_EMBEDDING_MODEL
        )
    except Exception as e:
        error_message = f"Error initializing OpenAIEmbeddings: {e}"
        print(error_message)
        return {"status": "error", "message": error_message, "retrieved_context": ""}

    print(f"Connecting to Pinecone index '{pinecone_index_name}'...")
    try:
        # PineconeVectorStore typically uses PINECONE_API_KEY and PINECONE_ENVIRONMENT 
        # from environment variables if not passed directly.
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index_name,
            embedding=embeddings
        )
        print("Successfully connected to Pinecone index.")
    except Exception as e:
        error_message = f"Error connecting to Pinecone index '{pinecone_index_name}': {e}"
        print(error_message)
        print("Ensure PINECONE_API_KEY and PINECONE_ENVIRONMENT are correctly set in your .env file,")
        print("and the index name matches an existing index in your Pinecone project.")
        return {"status": "error", "message": error_message, "retrieved_context": ""}

    print(f"Performing similarity search in Pinecone for query: '{user_query}'")
    try:
        retrieved_docs = vector_store.similarity_search(
            query=user_query,
            k=TOP_K_RESULTS
        )
    except Exception as e:
        error_message = f"Error during similarity search with Pinecone: {e}"
        print(error_message)
        return {"status": "error", "message": error_message, "retrieved_context": ""}

    if not retrieved_docs:
        print("No relevant documents found in Pinecone for the query.")
        return {"status": "success", "message": "No relevant information found in the knowledge base.", "retrieved_context": ""}

    print(f"Retrieved {len(retrieved_docs)} documents from Pinecone.")
    formatted_chunks_string = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    print("Successfully retrieved and formatted context from Pinecone.")
    return {"status": "success", "retrieved_context": formatted_chunks_string}

if __name__ == '__main__':
    # This example usage block requires the environment variables to be set correctly
    # and a Pinecone index populated by ingest_cloud.py to exist.
    print("Testing query_knowledge_base with Pinecone (ensure .env is set and index is populated)...")
    
    # Example: Load .env for direct script execution if not already loaded globally
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Reloading .env for __main__ test")
    #     load_dotenv() 
    
    test_query_pinecone = "What is a RAG model?"
    result_pinecone = query_knowledge_base(test_query_pinecone)
    
    print(f"\nQuery: {test_query_pinecone}")
    print(f"Status: {result_pinecone.get('status')}")
    if result_pinecone.get('message'):
        print(f"Message: {result_pinecone.get('message')}")
    if result_pinecone.get('retrieved_context'):
        print(f"Retrieved Context:\n{result_pinecone['retrieved_context']}")

    print("\n--- Test Finished ---")
    print("Important: For this test to work, ensure:")
    print("1. Your .env file in 'auhdhd_rag_agent' has OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME.")
    print("2. The specified Pinecone index exists and has been populated (e.g., by running ingest_cloud.py).")
