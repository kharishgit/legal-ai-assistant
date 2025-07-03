import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import requests
import json
from urllib.parse import quote
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from src.utils.logger import logger
from preprocessor import clean_text
from src.rag.vectorstore import initialize_vectorstore

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("IK_API_TOKEN")
if not API_TOKEN:
    logger.error("IK_API_TOKEN not found in .env file")
    sys.exit(1)

def fetch_cases(query, api_token, max_cases=15, max_pages=2):
    """Fetch cases from Indian Kanoon API using POST with query parameters."""
    docs = []
    for page in range(max_pages):
        encoded_query = quote(query)
        url = f"https://api.indiankanoon.org/search/?formInput={encoded_query}&pagenum={page}"
        headers = {"Authorization": f"Token {api_token}", "Accept": "application/json"}
        try:
            response = requests.post(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            debug_path = f"debug_api_response_{query.replace(' ', '_')}_page{page}.json"
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"API response for {query} page {page} saved to {debug_path}")
            page_docs = data.get("docs", [])
            if not page_docs:
                logger.warning(f"No documents found for {query} on page {page}")
                break
            docs.extend(page_docs)
            if len(docs) >= max_cases:
                break
        except Exception as e:
            logger.error(f"Failed to fetch {query} page {page}: {str(e)}")
            break
    return docs[:max_cases]

def fetch_full_doc(doc_id, api_token):
    """Fetch full document content from /doc/{tid}/ endpoint."""
    url = f"https://api.indiankanoon.org/doc/{doc_id}/"
    headers = {"Authorization": f"Token {api_token}", "Accept": "application/json"}
    try:
        response = requests.post(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        debug_path = f"debug_doc_{doc_id}.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Full doc for {doc_id} saved to {debug_path}")
        soup = BeautifulSoup(data.get("doc", ""), "html.parser")
        content = clean_text(soup.get_text(separator=" ", strip=True))
        return content
    except Exception as e:
        logger.error(f"Failed to fetch full doc for {doc_id}: {str(e)}")
        return None

def scrape_india_code(section):
    """Scrape statutory text from India Code."""
    output_dir = "data/raw/india_code"
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://www.indiacode.nic.in/handle/123456789/1362?view_type=search&sam_handle=123456789/1362&q={section}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        content = clean_text(soup.get_text(separator=" ", strip=True))
        if len(content) < 100:
            logger.warning(f"Insufficient content for {section}: {len(content)} chars")
            return None
        case_data = {
            "url": url,
            "title": f"Section {section} - India Code",
            "date": "Unknown",
            "content": content,
            "metadata": {
                "url": url,
                "case_id": f"indiacode_{section}",
                "court": "India Code",
                "date": "Unknown"
            }
        }
        json_path = os.path.join(output_dir, f"indiacode_{section}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved India Code data to {json_path}")
        return case_data
    except Exception as e:
        logger.error(f"Failed to scrape India Code for {section}: {str(e)}")
        return None

def scrape_ik_api(queries=["IPC 302", "Section 138 NI Act", "CrPC 482", "IPC 498A", "Article 21 Constitution"], max_cases=15, api_token=API_TOKEN):
    """Scrape cases via Indian Kanoon API and India Code."""
    output_dir = "data/raw/indkanoon_api"
    os.makedirs(output_dir, exist_ok=True)
    all_cases = []
    case_count = 0

    # Indian Kanoon cases
    for query in queries:
        if case_count >= 75:  # Limit to 75 cases (60 IK + 15 India Code)
            break
        docs = fetch_cases(query, api_token, max_cases)
        logger.info(f"Fetched {len(docs)} documents for {query}")
        
        for doc in docs:
            if case_count >= 75:
                break
            doc_id = str(doc.get("tid"))
            content = fetch_full_doc(doc_id, api_token)
            if not content or len(content) < 100:
                logger.warning(f"Insufficient content for {doc_id}: {len(content) if content else 0} chars")
                content = clean_text(doc.get("headline", ""))
                if len(content) < 100:
                    logger.warning(f"Insufficient headline content for {doc_id}: {len(content)} chars")
                    continue
            
            case_data = {
                "url": f"https://indiankanoon.org/doc/{doc_id}/",
                "title": doc.get("title", "Unknown Title"),
                "date": doc.get("publishdate", "Unknown Date"),
                "content": content,
                "metadata": {
                    "url": f"https://indiankanoon.org/doc/{doc_id}/",
                    "case_id": doc_id,
                    "court": doc.get("docsource", "Unknown"),
                    "date": doc.get("publishdate", "Unknown Date")
                }
            }
            
            json_filename = f"case_{case_count}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved case data to {json_path}")
            all_cases.append(case_data)
            case_count += 1

    # India Code statutes
    sections = ["302 IPC", "138 NI Act", "482 CrPC", "498A IPC", "Article 21 Constitution"]
    for section in sections:
        if case_count >= 75:
            break
        case_data = scrape_india_code(section)
        if case_data:
            json_path = os.path.join(output_dir, f"indiacode_{section.replace(' ', '_')}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved India Code data to {json_path}")
            all_cases.append(case_data)
            case_count += 1

    if all_cases:
        index_cases(all_cases)
    return all_cases

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
        raise

if __name__ == "__main__":
    cases = scrape_ik_api()
    if cases:
        print(f"Scraped {len(cases)} cases")
    else:
        print("No cases scraped. Check app.log and debug_api_response_*.json for details.")