import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Define constants
KNOWLEDGE_FILE = "knowledge.txt"
FAISS_INDEX_FILE = "faiss_index"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def ingest_documents():
    """
    Loads documents from the knowledge file, splits them into chunks,
    generates embeddings, and saves them to a FAISS vector store.
    """
    print("Starting document ingestion process...")

    # Check if knowledge file exists
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"Error: Knowledge file '{KNOWLEDGE_FILE}' not found.")
        return

    print("Loading documents...")
    loader = TextLoader(KNOWLEDGE_FILE)
    documents = loader.load()

    if not documents:
        print("No documents found in the knowledge file.")
        return

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    if not texts:
        print("Text splitting resulted in no chunks.")
        return

    print(f"Generating embeddings using '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        print(f"Error creating FAISS vector store: {e}")
        return

    print(f"Saving FAISS index to '{FAISS_INDEX_FILE}'...")
    try:
        vector_store.save_local(FAISS_INDEX_FILE)
        print("Vector store created and saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

if __name__ == "__main__":
    ingest_documents()
