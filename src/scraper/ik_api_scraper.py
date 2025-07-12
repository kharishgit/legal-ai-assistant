

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# import requests
# import json
# from urllib.parse import quote
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from src.utils.logger import logger
# from preprocessor import clean_text
# from src.rag.vectorstore import initialize_vectorstore

# # Load environment variables
# load_dotenv()
# API_TOKEN = os.getenv("IK_API_TOKEN")
# if not API_TOKEN:
#     logger.error("IK_API_TOKEN not found in .env file")
#     sys.exit(1)

# def fetch_cases(query, api_token, max_cases=15, max_pages=2):
#     """Fetch cases from Indian Kanoon API using POST with query parameters."""
#     docs = []
#     for page in range(max_pages):
#         encoded_query = quote(query)
#         url = f"https://api.indiankanoon.org/search/?formInput={encoded_query}&pagenum={page}"
#         headers = {"Authorization": f"Token {api_token}", "Accept": "application/json"}
#         try:
#             response = requests.post(url, headers=headers, timeout=30)
#             response.raise_for_status()
#             data = response.json()
#             debug_path = f"debug_api_response_{query.replace(' ', '_')}_page{page}.json"
#             with open(debug_path, "w", encoding="utf-8") as f:
#                 json.dump(data, f, ensure_ascii=False, indent=4)
#             logger.info(f"API response for {query} page {page} saved to {debug_path}")
#             page_docs = data.get("docs", [])
#             if not page_docs:
#                 logger.warning(f"No documents found for {query} on page {page}")
#                 break
#             docs.extend(page_docs)
#             if len(docs) >= max_cases:
#                 break
#         except Exception as e:
#             logger.error(f"Failed to fetch {query} page {page}: {str(e)}")
#             break
#     return docs[:max_cases]

# def fetch_full_doc(doc_id, api_token):
#     """Fetch full document content from /doc/{tid}/ endpoint."""
#     url = f"https://api.indiankanoon.org/doc/{doc_id}/"
#     headers = {"Authorization": f"Token {api_token}", "Accept": "application/json"}
#     try:
#         response = requests.post(url, headers=headers, timeout=30)
#         response.raise_for_status()
#         data = response.json()
#         debug_path = f"debug_doc_{doc_id}.json"
#         with open(debug_path, "w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=4)
#         logger.info(f"Full doc for {doc_id} saved to {debug_path}")
#         soup = BeautifulSoup(data.get("doc", ""), "html.parser")
#         content = clean_text(soup.get_text(separator=" ", strip=True))
#         return content
#     except Exception as e:
#         logger.error(f"Failed to fetch full doc for {doc_id}: {str(e)}")
#         return None

# def scrape_ik_api(queries=["IPC 302", "Section 138 NI Act", "CrPC 482", "IPC 498A"], max_cases=15, api_token=API_TOKEN):
#     """Scrape cases via Indian Kanoon API and save to JSON."""
#     output_dir = "data/raw/indkanoon_api"
#     os.makedirs(output_dir, exist_ok=True)
#     all_cases = []
#     case_count = 0

#     for query in queries:
#         if case_count >= 60:
#             break
#         docs = fetch_cases(query, api_token, max_cases)
#         logger.info(f"Fetched {len(docs)} documents for {query}")
        
#         for doc in docs:
#             if case_count >= 60:
#                 break
#             doc_id = str(doc.get("tid"))  # Ensure string ID
#             content = fetch_full_doc(doc_id, api_token)
#             if not content or len(content) < 100:
#                 logger.warning(f"Insufficient content for {doc_id}: {len(content) if content else 0} chars")
#                 content = clean_text(doc.get("headline", ""))  # Fallback to headline
#                 if len(content) < 100:
#                     logger.warning(f"Insufficient headline content for {doc_id}: {len(content)} chars")
#                     continue
            
#             case_data = {
#                 "url": f"https://indiankanoon.org/doc/{doc_id}/",
#                 "title": doc.get("title", "Unknown Title"),
#                 "date": doc.get("publishdate", "Unknown Date"),
#                 "content": content,
#                 "metadata": {
#                     "url": f"https://indiankanoon.org/doc/{doc_id}/",
#                     "case_id": doc_id,
#                     "court": doc.get("docsource", "Unknown"),
#                     "date": doc.get("publishdate", "Unknown Date")
#                 }
#             }
            
#             json_filename = f"case_{case_count}.json"
#             json_path = os.path.join(output_dir, json_filename)
#             with open(json_path, "w", encoding="utf-8") as f:
#                 json.dump(case_data, f, ensure_ascii=False, indent=4)
#             logger.info(f"Saved case data to {json_path}")
#             all_cases.append(case_data)
#             case_count += 1

#     if all_cases:
#         index_cases(all_cases)
#     return all_cases

# def index_cases(cases):
#     """Index cases in ChromaDB."""
#     try:
#         vectorstore = initialize_vectorstore()
#         texts = [case["content"] for case in cases]
#         metadatas = [case["metadata"] for case in cases]
#         ids = [case["metadata"]["case_id"] for case in cases]
#         vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
#         logger.info(f"Indexed {len(cases)} cases in ChromaDB")
#     except Exception as e:
#         logger.error(f"Failed to index cases: {str(e)}")
#         raise

# if __name__ == "__main__":
#     cases = scrape_ik_api()
#     if cases:
#         print(f"Scraped {len(cases)} cases")
#     else:
#         print("No cases scraped. Check app.log and debug_api_response_*.json for details.")


import requests
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import logger
import re

load_dotenv()
IK_API_TOKEN = os.getenv("IK_API_TOKEN")
PERSIST_DIRECTORY = "data/vector_db"

def extract_sections(text):
    """Extract IPC/CrPC/NI Act sections from text."""
    sections = []
    # Simple regex for IPC, CrPC, NI Act sections
    patterns = [r"Section (\d+[A-Z]?) of the IPC", r"Section (\d+[A-Z]?) of the CrPC", r"Section (\d+[A-Z]?) of the NI Act"]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        sections.extend([f"{match} {pattern.split()[-1]}" for match in matches])
    return sections

def scrape_ik_data(search_query, max_results=10):
    """Scrape and index case data from Indian Kanoon API."""
    try:
        headers = {"Authorization": f"Token {IK_API_TOKEN}"}
        params = {
            "q": search_query,
            "max_results": max_results,
            "sort_by": "relevance"
        }
        response = requests.get(
            "https://api.indiankanoon.org/search/",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        data = response.json()
        documents = []
        for doc in data.get("docs", []):
            text = doc.get("text", "")
            sections = extract_sections(text)
            metadata = {
                "case_id": doc.get("id", ""),
                "court": doc.get("court_name", ""),
                "date": doc.get("date", ""),
                "url": doc.get("url", ""),
                "sections": sections or ["unknown"]
            }
            documents.append({
                "text": text,
                "metadata": metadata
            })
        logger.info(f"Scraped {len(documents)} documents for query: {search_query}")
        return documents
    except Exception as e:
        logger.error(f"Failed to scrape Indian Kanoon data: {str(e)}")
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
    queries = ["IPC 302", "CrPC 482", "IPC 498A", "Section 138 NI Act"]
    for query in queries:
        docs = scrape_ik_data(query, max_results=20)
        if docs:
            index_documents(docs)