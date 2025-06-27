import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import requests
import PyPDF2
import json
import time
from src.utils.logger import logger
from preprocessor import clean_text
from src.rag.vectorstore import initialize_vectorstore
from pathlib import Path
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright, TimeoutError

def scrape_supreme_court(max_cases=50):
    """Scrape Supreme Court judgments and index in ChromaDB."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)  # Bypass SSL errors
        page = context.new_page()
        cases = []
        case_count = 0

        url = "https://main.sci.gov.in/judgments"
        logger.info(f"Scraping Supreme Court judgments: {url}")
        
        for attempt in range(3):
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_selector("a[href*='.pdf']", timeout=30000)
                pdf_links = page.evaluate('''() => {
                    return Array.from(document.querySelectorAll("a[href*='.pdf']")).map(el => el.href);
                }''')[:max_cases]
                logger.info(f"Extracted {len(pdf_links)} PDF links")
                
                for pdf_url in pdf_links:
                    if case_count >= max_cases:
                        break
                    case_data = scrape_pdf(pdf_url)
                    if case_data:
                        cases.append(case_data)
                        save_case(case_data)
                        case_count += 1
                        logger.info(f"Scraped case {case_count}/{max_cases}: {case_data['title']}")
                    time.sleep(1)  # Rate limit
                break
            except TimeoutError:
                logger.error(f"Attempt {attempt+1} timed out for {url}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed for {url}: {str(e)}")
                time.sleep(5)
        
        context.close()
        browser.close()
        if cases:
            index_cases(cases)
        logger.info(f"Total cases scraped: {len(cases)}")
        return cases

def scrape_pdf(pdf_url):
    """Download and extract text from PDF."""
    try:
        response = requests.get(pdf_url, timeout=30, verify=False)  # Bypass SSL verification
        response.raise_for_status()
        pdf_path = f"debug_pdf_{pdf_url.split('/')[-1]}"
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            content_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content_text += text
            content_text = clean_text(content_text)
        
        if not content_text or len(content_text) < 20:
            logger.warning(f"Insufficient content for {pdf_url}: {len(content_text)} chars")
            return None
        
        case_id = pdf_url.split("/")[-1].replace(".pdf", "")
        return {
            "title": f"Supreme Court Judgment {case_id}",
            "content": content_text,
            "metadata": {
                "url": pdf_url,
                "case_id": case_id,
                "court": "Supreme Court of India",
                "date": "Unknown"  # Extract from PDF if possible
            }
        }
    except Exception as e:
        logger.error(f"Failed to scrape PDF {pdf_url}: {str(e)}")
        return None

def save_case(case_data):
    """Save case data to JSON file."""
    case_id = case_data["metadata"]["case_id"]
    output_path = Path(f"data/raw/sci/case_{case_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(case_data, f, indent=2)
    logger.info(f"Saved case to {output_path}")

def index_cases(cases):
    """Index cases in ChromaDB."""
    try:
        vectorstore = initialize_vectorstore()
        texts = [case["content"] for case in cases]
        metadatas = [case["metadata"] for case in cases]
        ids = [case["metadata"]["case_id"] for case in cases]
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Indexed {len(cases)} cases in ChromaDB")
    except Exception as e:
        logger.error(f"Failed to index cases: {str(e)}")

if __name__ == "__main__":
    scrape_supreme_court(max_cases=50)