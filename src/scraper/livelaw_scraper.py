import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



from playwright.sync_api import sync_playwright
import os
from src.utils.logger import logger
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re

PERSIST_DIRECTORY = "data/vector_db"

def extract_sections(text):
    """Extract IPC/CrPC/NI Act sections from text."""
    try:
        sections = []
        patterns = [
            r"Section\s+(\d+[A-Z]?)[\s]*(?:of the)?\s*(IPC|CrPC|NI Act)",
            r"\b(\d+[A-Z]?)\s*(IPC|CrPC|NI Act)\b"
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend([f"{match[0]} {match[1]}" for match in matches])
        return list(set(sections)) or ["unknown"]
    except Exception as e:
        logger.error(f"Failed to extract sections: {str(e)}")
        return ["unknown"]

def scrape_livelaw(max_articles=20):
    """Scrape recent articles from LiveLaw using Playwright."""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()
            page.goto("https://www.livelaw.in/news-updates/", timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)
            articles = page.query_selector_all("div.post-item")
            documents = []
            for i, article in enumerate(articles[:max_articles]):
                title = article.query_selector("h2").text_content().strip() if article.query_selector("h2") else ""
                content = article.query_selector("div.post-excerpt").text_content().strip() if article.query_selector("div.post-excerpt") else ""
                link = article.query_selector("a").get_attribute("href") if article.query_selector("a") else ""
                sections = extract_sections(title + " " + content)
                metadata = {
                    "case_id": f"livelaw_{i}_{hashlib.md5(title.encode()).hexdigest()}",
                    "court": "unknown",
                    "date": "2025-07-14",
                    "url": link,
                    "sections": sections
                }
                documents.append({
                    "text": f"{title}\n{content}",
                    "metadata": metadata
                })
            logger.info(f"Scraped {len(documents)} articles from LiveLaw")
            browser.close()
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
        texts = [doc["text"] for doc in documents if doc["text"].strip()]
        metadatas = [doc["metadata"] for doc in documents if doc["text"].strip()]
        ids = [doc["metadata"]["case_id"] for doc in documents if doc["text"].strip()]
        if texts:
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(texts)} documents into ChromaDB")
        else:
            logger.warning("No valid documents to index")
    except Exception as e:
        logger.error(f"Failed to index documents: {str(e)}")

if __name__ == "__main__":
    docs = scrape_livelaw(max_articles=20)
    if docs:
        index_documents(docs)
