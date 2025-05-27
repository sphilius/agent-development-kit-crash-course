import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone as PineconeClient, ServerRelativeSpec # For index management

# Define constants
KNOWLEDGE_FILE_PATH = "auhdhd_rag_agent/knowledge.txt"
# OpenAI's text-embedding-ada-002 model dimension
OPENAI_EMBEDDING_DIMENSION = 1536
# Pinecone spec - this might need to be adjusted based on the user's Pinecone setup
# Common choices: 'aws', 'gcp', 'azure'. Region also needs to be valid for the cloud.
# For starter/free tiers, it's often a specific environment like "gcp-starter" or "us-west1-gcp"
# For serverless, it would be ServerlessSpec(cloud='aws', region='us-west-2') for example.
# Using a common default, but user might need to change this.
PINECONE_CLOUD_PROVIDER = os.getenv("PINECONE_CLOUD_PROVIDER", "aws") # Default to aws
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # Default to us-east-1

def ingest_documents_to_pinecone():
    """
    Loads documents, splits them, generates embeddings using OpenAI,
    and upserts them to a Pinecone vector store.
    """
    print("Starting document ingestion process for Pinecone...")

    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT") # This is the project environment/region for Pinecone
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    # Validate environment variables
    if not all([openai_api_key, pinecone_api_key, pinecone_environment, pinecone_index_name]):
        print("Error: Missing one or more required environment variables:")
        if not openai_api_key: print("- OPENAI_API_KEY")
        if not pinecone_api_key: print("- PINECONE_API_KEY")
        if not pinecone_environment: print("- PINECONE_ENVIRONMENT (e.g., 'gcp-starter', 'us-east-1')")
        if not pinecone_index_name: print("- PINECONE_INDEX_NAME")
        return

    print("Initializing Pinecone client...")
    try:
        pc = PineconeClient(api_key=pinecone_api_key) # environment is not passed here for v3.x client
    except Exception as e:
        print(f"Error initializing Pinecone client: {e}")
        return

    # Check if the Pinecone index exists, create if not
    print(f"Checking for Pinecone index '{pinecone_index_name}'...")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if pinecone_index_name not in existing_indexes:
        print(f"Index '{pinecone_index_name}' not found. Creating new index...")
        try:
            # For Pinecone serverless, the dimension is not specified during index creation with a Pod-based spec.
            # The spec parameter is crucial and depends on the type of Pinecone index (serverless vs. pod-based).
            # ServerRelativeSpec is for pod-based indexes.
            # For serverless, it would be: from pinecone import ServerlessSpec
            # spec = ServerlessSpec(cloud=PINECONE_CLOUD_PROVIDER, region=PINECONE_REGION)
            # For this example, assuming a pod-based index for now.
            pc.create_index(
                name=pinecone_index_name,
                dimension=OPENAI_EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerRelativeSpec(cloud=PINECONE_CLOUD_PROVIDER, region=PINECONE_REGION)
            )
            print(f"Pinecone index '{pinecone_index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating Pinecone index: {e}")
            print("Please ensure your Pinecone environment and index specifications (cloud, region) are correct.")
            print("For Pinecone free tier or serverless, the 'spec' might differ.")
            print("Example for serverless: spec=ServerlessSpec(cloud='aws', region='us-west-2')")
            return
    else:
        print(f"Pinecone index '{pinecone_index_name}' already exists.")

    # Initialize OpenAI Embeddings
    print("Initializing OpenAI embeddings model...")
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
    except Exception as e:
        print(f"Error initializing OpenAIEmbeddings: {e}")
        return

    # Load and split documents
    print(f"Loading documents from '{KNOWLEDGE_FILE_PATH}'...")
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"Error: Knowledge file '{KNOWLEDGE_FILE_PATH}' not found.")
        return
    try:
        loader = TextLoader(file_path=KNOWLEDGE_FILE_PATH, encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    if not documents:
        print("No documents found in the knowledge file.")
        return

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    if not texts:
        print("Text splitting resulted in no chunks.")
        return
    
    print(f"Found {len(texts)} text chunks to embed and upsert.")

    # Generate embeddings and upsert to Pinecone
    print(f"Generating embeddings and upserting to Pinecone index '{pinecone_index_name}'...")
    try:
        PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=pinecone_index_name
            # namespace can be added here if needed: namespace="your_namespace"
        )
        print("Documents successfully ingested into Pinecone.")
    except Exception as e:
        print(f"Error during embedding generation or upserting to Pinecone: {e}")
        print("Ensure your OpenAI API key has embedding permissions and your Pinecone index is configured correctly.")

if __name__ == "__main__":
    ingest_documents_to_pinecone()
    print("\n--- Ingestion Script Finished ---")
    print("Important Reminders:")
    print("1. Ensure your .env file has correct OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME.")
    print("2. Your OpenAI API key must have embedding permissions.")
    print("3. Your Pinecone index name should be unique and correctly specified.")
    print("4. The Pinecone index 'spec' (cloud, region) in the script might need adjustment based on your Pinecone setup (e.g., free tier, serverless, specific cloud provider).")
    print("   The current default is ServerRelativeSpec(cloud='aws', region='us-east-1').")
