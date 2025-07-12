import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


import requests
from bs4 import BeautifulSoup
from src.utils.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

PERSIST_DIRECTORY = "data/vector_db"

def scrape_livelaw(max_articles=10):
    """Scrape recent articles from LiveLaw."""
    try:
        url = "https://www.livelaw.in/news-updates/"
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("div", class_="post-item", limit=max_articles)
        documents = []
        for i, article in enumerate(articles):
            title = article.find("h2").text.strip() if article.find("h2") else ""
            content = article.find("div", class_="post-excerpt").text.strip() if article.find("div", class_="post-excerpt") else ""
            link = article.find("a")["href"] if article.find("a") else ""
            sections = extract_sections(content + " " + title)  # Reuse from ik_api_scraper
            metadata = {
                "case_id": f"livelaw_{i}",
                "court": "unknown",
                "date": "2025-07-12",
                "url": link,
                "sections": sections or ["unknown"]
            }
            documents.append({
                "text": f"{title}\n{content}",
                "metadata": metadata
            })
        logger.info(f"Scraped {len(documents)} articles from LiveLaw")
        return documents
    except Exception as e:
        logger.error(f"Failed to scrape LiveLaw: {str(e)}")
        return []

def index_documents(documents):
    """Index documents into ChromaDB."""
    try:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        )
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [doc["metadata"]["case_id"] for doc in documents]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Indexed {len(documents)} documents into ChromaDB")
    except Exception as e:
        logger.error(f"Failed to index documents: {str(e)}")

if __name__ == "__main__":
    docs = scrape_livelaw(max_articles=20)
    if docs:
        index_documents(docs)