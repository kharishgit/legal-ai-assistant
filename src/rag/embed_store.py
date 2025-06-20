
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer
import json
from src.utils.logger import logger

def create_embeddings(input_file="data/processed/combined_dataset.json", collection_name="legal_cases"):
    """
    Generate embeddings for dataset and store in ChromaDB.
    """
    try:
        logger.info(f"Loading dataset from {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)

        # Initialize embedding model
        logger.info("Initializing sentence transformer model")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB client")
        client = chromadb.PersistentClient(path="data/vector_db")
        # Delete and recreate collection for fresh start
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name)

        # Generate and store embeddings
        logger.info("Generating embeddings")
        valid_entries = 0
        for i, entry in enumerate(data):
            text = entry.get("text", "")
            if not text or "No content found" in text:
                logger.warning(f"Skipping entry {i}: Invalid or empty text")
                continue
            embedding = model.encode(text).tolist()
            metadata = entry.get("metadata", {})
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"doc_{i}"]
            )
            valid_entries += 1
        logger.info(f"Stored {valid_entries} documents in ChromaDB")
        return valid_entries
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise

def test_retrieval(query, collection_name="legal_cases", top_k=3):
    """
    Test retrieval from ChromaDB with a sample query.
    """
    try:
        logger.info(f"Testing retrieval for query: {query}")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="data/vector_db")
        collection = client.get_collection(name=collection_name)
        query_embedding = model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        logger.info(f"Top {top_k} results: {results['documents']}")
        return results
    except Exception as e:
        logger.error(f"Error in test_retrieval: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        create_embeddings()
        test_retrieval("What is the punishment for murder under IPC 302?")
        test_retrieval("Summarize a case related to IPC 302")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")