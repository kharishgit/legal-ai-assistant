
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import chromadb
from sentence_transformers import SentenceTransformer
import json
from src.utils.logger import logger
from src.utils.text_processing import chunk_text

def create_embeddings(input_file="data/processed/combined_dataset.json", collection_name="legal_cases"):
    try:
        logger.info(f"Loading dataset from {input_file}")
        with open(input_file, 'r') as f:
            data = json.load(f)

        logger.info("Initializing sentence transformer model")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initializing ChromaDB client")
        client = chromadb.PersistentClient(path="data/vector_db")
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name)

        logger.info("Generating embeddings")
        valid_entries = 0
        for i, entry in enumerate(data):
            text = entry.get("text", "")
            if not text or "No content found" in text:
                logger.warning(f"Skipping entry {i}: Invalid or empty text")
                continue
            # Chunk long texts
            chunks = chunk_text(text)
            for j, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()
                metadata = entry.get("metadata", {})
                metadata["chunk_id"] = f"{i}_{j}"
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"doc_{i}_{j}"]
                )
                valid_entries += 1
        logger.info(f"Stored {valid_entries} documents in ChromaDB")
        return valid_entries
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        raise

def test_retrieval(query, collection_name="legal_cases", top_k=3):
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