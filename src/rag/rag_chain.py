


# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# import re
# import logging
# import pickle
# import hashlib
# from typing import List, Dict, Optional
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from src.rag.vectorstore import initialize_vectorstore
# from src.data.update_dataset import update_dataset
# from src.utils.logger import logger
# import pytesseract
# from PIL import Image

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# # Caching setup
# CACHE_DIR = "data/cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# def get_cache_key(query: str) -> str:
#     """Generate cache key from query."""
#     return hashlib.md5(query.encode()).hexdigest()

# def load_from_cache(query: str) -> Optional[Dict]:
#     """Load cached result for query."""
#     cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             logger.info(f"Loaded cached result for query: {query[:50]}...")
#             return pickle.load(f)
#     return None

# def save_to_cache(query: str, result: Dict) -> None:
#     """Save result to cache."""
#     cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
#     with open(cache_file, "wb") as f:
#         pickle.dump(result, f)
#     logger.info(f"Saved result to cache for query: {query[:50]}...")

# def extract_legal_sections(text: str) -> List[str]:
#     """Extract legal sections (e.g., IPC 420, CrPC 482) from query or notice text."""
#     patterns = [
#         r'IPC\s*\d+[A-Za-z]?',
#         r'Section\s*\d+[A-Za-z]?\s*(?:of\s*(?:Negotiable\s*Instruments\s*Act|NI\s*Act))',
#         r'CrPC\s*\d+[A-Za-z]?',
#         r'Article\s*\d+[A-Za-z]?'
#     ]
#     sections = []
#     for pattern in patterns:
#         matches = re.findall(pattern, text, re.IGNORECASE)
#         sections.extend(matches)
#     return [s.strip() for s in sections]

# def process_notice_image(image_path: str) -> Dict:
#     """Process a scanned notice image to extract text and metadata."""
#     try:
#         image = Image.open(image_path)
#         text = pytesseract.image_to_string(image, lang='eng')
#         logger.info(f"Extracted text from image: {image_path}")
        
#         # Extract metadata
#         case_number = re.search(r'(CRL|WP|CA|MA)/\d+/\d{4}', text, re.IGNORECASE)
#         sections = extract_legal_sections(text)
#         metadata = {
#             "case_number": case_number.group(0) if case_number else "Unknown",
#             "sections": sections,
#             "source": "Notice"
#         }
#         return {"text": text, "metadata": metadata}
#     except Exception as e:
#         logger.error(f"Failed to process image {image_path}: {e}")
#         return {"text": "", "metadata": {}}

# def summarize_context(docs: List, max_tokens: int = 2000) -> str:
#     """Summarize and truncate documents to fit token limit."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100,
#         length_function=len
#     )
#     valid_texts = [doc.page_content for doc in docs if hasattr(doc, 'page_content') and doc.page_content.strip()]
#     context = " ".join(valid_texts)
#     chunks = text_splitter.split_text(context)
#     summarized = "\n".join(chunks[:3])  # Limit to 3 chunks (~1500 tokens)
#     if len(summarized) > max_tokens:
#         summarized = summarized[:max_tokens] + "..."
#     logger.info(f"Summarized context to {len(summarized)} characters")
#     return summarized

# def run_rag_chain(query: str, image_path: Optional[str] = None) -> Dict:
#     """Run RAG pipeline for query or notice analysis."""
#     # Check cache
#     cache_key = query + (image_path if image_path else "")
#     cached_result = load_from_cache(cache_key)
#     if cached_result:
#         return cached_result

#     vectorstore = initialize_vectorstore()
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=512)

#     # Process notice if provided
#     notice_data = {}
#     if image_path:
#         notice_data = process_notice_image(image_path)
#         query = f"{query} {notice_data['text']}" if notice_data.get("text") else query

#     # Extract legal sections and trigger dataset update
#     sections = extract_legal_sections(query)
#     if sections:
#         logger.info(f"Extracted sections: {sections}")
#         update_dataset(queries=sections)
#         logger.info(f"Updated dataset with sections: {sections}")

#     # Query ChromaDB with metadata filtering
#     try:
#         results = vectorstore.similarity_search(
#             query,
#             k=5,
#             filter={"query": {"$in": sections}} if sections else None
#         )
#         logger.info(f"Retrieved {len(results)} documents from ChromaDB")
#     except Exception as e:
#         logger.error(f"ChromaDB query failed: {e}")
#         results = []

#     # Summarize context
#     context = summarize_context(results, max_tokens=2000)
#     metadata = [{"case_id": doc.metadata.get("case_id", "Unknown"), 
#                  "court": doc.metadata.get("court", "Unknown"), 
#                  "date": doc.metadata.get("date", "Unknown")} for doc in results]

#     # Define prompt
#     prompt_template = PromptTemplate(
#         input_variables=["query", "context", "metadata"],
#         template="""You are a legal assistant for Indian law. Based on the query and context from Indian Kanoon cases, provide a concise response:
#         - For statutory questions, quote the law and explain in 2-3 simple sentences.
#         - For case summaries, provide 3 bullet points (key facts, court, outcome).
#         - For notices, explain legal implications, 2-3 next steps, 1-2 opponent arguments with counterpoints.
#         If no Indian precedents, suggest checking US/EU cases. Use simple language.

# Query: {query}
# Context: {context}
# Metadata: {metadata}
# """
#     )

#     # Generate response
#     try:
#         chain = prompt_template | llm
#         response = chain.invoke({"query": query, "context": context, "metadata": metadata})
#         result = {
#             "response": response.content,
#             "metadata": metadata,
#             "notice_data": notice_data if image_path else None
#         }
#         save_to_cache(cache_key, result)
#         logger.info("Generated and cached response for query")
#         return result
#     except Exception as e:
#         logger.error(f"Failed to generate response: {e}")
#         return {"response": "Error processing query", "metadata": [], "notice_data": notice_data}

# if __name__ == "__main__":
#     # Test with a lawyer query
#     lawyer_query = "Latest Supreme Court judgments on Section 138 NI Act"
#     result = run_rag_chain(lawyer_query)
#     print(f"Lawyer Query Response: {result['response']}")

#     # Test with a layman query and notice
#     layman_query = "My landlord is not returning my deposit"
#     notice_path = "data/sample_notice.png"
#     result = run_rag_chain(layman_query, notice_path)
#     print(f"Layman Query Response: {result['response']}")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
import logging
import pickle
import hashlib
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.rag.vectorstore import initialize_vectorstore
from src.data.update_dataset import update_dataset
from src.utils.logger import logger
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Caching setup
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return hashlib.md5(query.encode()).hexdigest()

def load_from_cache(query: str) -> Optional[Dict]:
    """Load cached result for query."""
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            logger.info(f"Loaded cached result for query: {query[:50]}...")
            return pickle.load(f)
    return None

def save_to_cache(query: str, result: Dict) -> None:
    """Save result to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Saved result to cache for query: {query[:50]}...")

def extract_legal_sections(text: str) -> List[str]:
    """Extract legal sections (e.g., IPC 420, CrPC 482, Section 54) from query or notice text."""
    patterns = [
        r'IPC\s*\d+[A-Za-z]?',
        r'Section\s*\d+[A-Za-z]?\s*(?:of\s*(?:Negotiable\s*Instruments\s*Act|NI\s*Act|Transfer\s*of\s*Property\s*Act))?',
        r'CrPC\s*\d+[A-Za-z]?',
        r'Article\s*\d+[A-Za-z]?'
    ]
    sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        sections.extend(matches)
    return [s.strip() for s in sections]

def process_notice_image(image_path: str) -> Dict:
    """Process a scanned notice image to extract text and metadata."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang='eng')
        logger.info(f"Extracted text from image: {image_path}")
        
        # Extract metadata
        case_number = re.search(r'(CRL|WP|CA|CIV|MA)/\d+/\d{4}', text, re.IGNORECASE)
        sections = extract_legal_sections(text)
        metadata = {
            "case_number": case_number.group(0) if case_number else "Unknown",
            "sections": sections,
            "source": "Notice"
        }
        return {"text": text, "metadata": metadata}
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        return {"text": "", "metadata": {}}

def summarize_context(docs: List, max_tokens: int = 2000) -> str:
    """Summarize and truncate documents to fit token limit."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    valid_texts = [doc.page_content for doc in docs if hasattr(doc, 'page_content') and doc.page_content.strip()]
    context = " ".join(valid_texts)
    chunks = text_splitter.split_text(context)
    summarized = "\n".join(chunks[:3])  # Limit to 3 chunks (~1500 tokens)
    if len(summarized) > max_tokens:
        summarized = summarized[:max_tokens] + "..."
    logger.info(f"Summarized context to {len(summarized)} characters")
    return summarized

def run_rag_chain(query: str, image_path: Optional[str] = None) -> Dict:
    """Run RAG pipeline for query or notice analysis."""
    # Check cache
    cache_key = query + (image_path if image_path else "")
    cached_result = load_from_cache(cache_key)
    if cached_result:
        return cached_result

    vectorstore = initialize_vectorstore()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, max_tokens=512)

    # Process notice if provided
    notice_data = {}
    if image_path:
        notice_data = process_notice_image(image_path)
        query = f"{query} {notice_data['text']}" if notice_data.get("text") else query

    # Extract legal sections and trigger dataset update
    sections = extract_legal_sections(query)
    if sections:
        logger.info(f"Extracted sections: {sections}")
        update_dataset(queries=sections)
        logger.info(f"Updated dataset with sections: {sections}")

    # Query ChromaDB with metadata filtering
    try:
        results = vectorstore.similarity_search(
            query,
            k=5,
            filter={"query": {"$in": sections}} if sections else None
        )
        logger.info(f"Retrieved {len(results)} documents from ChromaDB")
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        results = []

    # Summarize context
    context = summarize_context(results, max_tokens=2000)
    metadata = [{"case_id": doc.metadata.get("case_id", "Unknown"), 
                 "court": doc.metadata.get("court", "Unknown"), 
                 "date": doc.metadata.get("date", "Unknown")} for doc in results]

    # Define prompt with enhanced opponent arguments
    prompt_template = PromptTemplate(
        input_variables=["query", "context", "metadata"],
        template="""You are a legal assistant for Indian law. Based on the query and context from Indian Kanoon cases, provide a concise response:
        - For statutory questions, quote the law and explain in 2-3 simple sentences.
        - For case summaries, provide 3 bullet points (key facts, court, outcome).
        - For notices, explain legal implications, 2-3 next steps, and 1-2 opponent arguments with counterpoints (e.g., opponent may claim valid property transfer; counter with evidence of fraud or non-compliance).
        If no Indian precedents, suggest checking US/EU cases. Use simple language.

Query: {query}
Context: {context}
Metadata: {metadata}
"""
    )

    # Generate response
    try:
        chain = prompt_template | llm
        response = chain.invoke({"query": query, "context": context, "metadata": metadata})
        result = {
            "response": response.content,
            "metadata": metadata,
            "notice_data": notice_data if image_path else None
        }
        save_to_cache(cache_key, result)
        logger.info("Generated and cached response for query")
        return result
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return {"response": "Error processing query", "metadata": [], "notice_data": notice_data}

if __name__ == "__main__":
    # Test with a lawyer query
    lawyer_query = "Latest Supreme Court judgments on Section 138 NI Act"
    result = run_rag_chain(lawyer_query)
    print(f"Lawyer Query Response: {result['response']}")

    # Test with a layman query and notice
    layman_query = "My landlord is not returning my deposit"
    notice_path = "data/sample_notice.png"
    result = run_rag_chain(layman_query, notice_path)
    print(f"Layman Query Response: {result['response']}")