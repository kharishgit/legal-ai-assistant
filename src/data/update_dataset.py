import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import os
from src.scraper.ik_api_scraper import scrape_ik_data, index_documents
from src.scraper.livelaw_scraper import scrape_livelaw
from src.utils.logger import logger

DATASET_PATH = "data/processed/combined_dataset.json"

def load_existing_dataset():
    """Load existing combined_dataset.json."""
    try:
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, "r") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return []

def save_dataset(dataset):
    """Save dataset to combined_dataset.json."""
    try:
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, "w") as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Saved {len(dataset)} documents to {DATASET_PATH}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")

def update_dataset():
    """Update dataset with new scraped data."""
    try:
        existing_dataset = load_existing_dataset()
        existing_ids = {doc["metadata"]["case_id"] for doc in existing_dataset}
        
        # Scrape new data
        queries = ["IPC 302", "CrPC 482", "IPC 498A", "Section 138 NI Act", "IPC 376", "Article 14"]
        ik_docs = []
        for query in queries:
            docs = scrape_ik_data(query, max_results=20)
            ik_docs.extend([doc for doc in docs if doc["metadata"]["case_id"] not in existing_ids])
        
        livelaw_docs = scrape_livelaw(max_articles=20)
        livelaw_docs = [doc for doc in livelaw_docs if doc["metadata"]["case_id"] not in existing_ids]
        
        # Combine and save
        new_dataset = existing_dataset + ik_docs + livelaw_docs
        save_dataset(new_dataset)
        
        # Index to vectorstore
        all_docs = ik_docs + livelaw_docs
        if all_docs:
            index_documents(all_docs)
        
        logger.info(f"Added {len(ik_docs)} Indian Kanoon and {len(livelaw_docs)} LiveLaw documents")
    except Exception as e:
        logger.error(f"Failed to update dataset: {str(e)}")

if __name__ == "__main__":
    update_dataset()
