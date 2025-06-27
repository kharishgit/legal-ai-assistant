from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def initialize_vectorstore():
    """Initialize ChromaDB with sentence-transformer embeddings."""
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Use MPS later if needed
    )
    return Chroma(
        persist_directory="data/vector_db",
        embedding_function=embedding_function,
        collection_name="legal_cases"
    )