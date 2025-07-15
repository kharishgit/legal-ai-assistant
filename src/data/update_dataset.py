import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import logging
import argparse
from typing import List, Dict, Optional
from src.scraper.ik_api_scraper import scrape_ik_api
from src.rag.vectorstore import initialize_vectorstore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"
OUTPUT_FILE = os.path.join(DATA_DIR, "combined_dataset.json")

def update_dataset(queries: Optional[List[str]] = None, max_cases_per_query: int = 15) -> None:
    """Update combined_dataset.json with new cases from Indian Kanoon and index in ChromaDB."""
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_docs = []
    
    # Load existing dataset
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_docs = json.load(f)
            logger.info(f"Loaded {len(existing_docs)} existing documents from {OUTPUT_FILE}")
        except Exception as e:
            logger.error(f"Failed to load existing dataset: {e}")

    # Default queries if none provided
    default_queries = ["IPC 302", "Section 138 NI Act", "CrPC 482", "IPC 498A"]
    queries = queries or default_queries

    # Fetch new cases
    new_docs = []
    for query in queries:
        try:
            cases = scrape_ik_api(queries=[query], max_cases=max_cases_per_query)
            new_docs.extend(cases)
            logger.info(f"Fetched {len(cases)} new cases for query: {query}")
        except Exception as e:
            logger.error(f"Failed to fetch cases for {query}: {e}")

    # Combine and deduplicate
    combined_docs = existing_docs + new_docs
    unique_docs = []
    seen_ids = set()
    for doc in combined_docs:
        case_id = doc["metadata"].get("case_id")
        if case_id and case_id not in seen_ids:
            seen_ids.add(case_id)
            unique_docs.append(doc)
    
    # Save updated dataset
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(unique_docs, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(unique_docs)} documents to {OUTPUT_FILE}")
        logger.info(f"Added {len(new_docs)} new Indian Kanoon documents")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

    # Index in ChromaDB
    if new_docs:
        try:
            vectorstore = initialize_vectorstore()
            texts = [doc["content"] for doc in new_docs]
            metadatas = [doc["metadata"] for doc in new_docs]
            ids = [doc["metadata"]["case_id"] for doc in new_docs]
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(new_docs)} new cases in ChromaDB")
        except Exception as e:
            logger.error(f"Failed to index new cases: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update legal dataset with Indian Kanoon cases")
    parser.add_argument("--queries", nargs="*", default=None, help="List of queries (e.g., 'IPC 420' 'Article 32')")
    args = parser.parse_args()
    update_dataset(queries=args.queries)