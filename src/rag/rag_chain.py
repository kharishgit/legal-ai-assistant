import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# import os
# import time
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from src.ocr.ocr_processor import process_notice
# from src.utils.logger import logger
# import pickle
# import hashlib
# import sys
# import json

# # Add parent directory to sys.path for module imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# # Disable HuggingFace tokenizer parallelism
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not found in .env file")

# # Simple caching mechanism
# CACHE_DIR = "data/cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# def get_cache_key(query):
#     return hashlib.md5(query.encode()).hexdigest()

# def load_from_cache(query):
#     cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
#     if os.path.exists(cache_file):
#         with open(cache_file, "rb") as f:
#             return pickle.load(f)
#     return None

# def save_to_cache(query, result):
#     cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
#     with open(cache_file, "wb") as f:
#         pickle.dump(result, f)

# def initialize_rag_chain():
#     try:
#         # Initialize text splitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             length_function=len
#         )

#         # Initialize vectorstore
#         vectorstore = Chroma(
#             persist_directory="data/vector_db",
#             embedding_function=HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-mpnet-base-v2",
#                 model_kwargs={"device": "cpu"}
#             )
#         )
#         logger.info("Initialized ChromaDB for RAG")

        
#         def get_relevant_documents(input_data):
#             question = input_data["question"]
#             metadata = input_data.get("metadata", {})
#             search_kwargs = {"k": 5, "fetch_k": 20}  # Increase for better recall
#             if metadata and "sections" in metadata:
#                 search_kwargs["filter"] = {"sections": {"$contains": metadata["sections"][0]}}  # Use $contains for partial matches
#             retriever = vectorstore.as_retriever(
#                 search_type="similarity",  # Switch to similarity for broader results
#                 search_kwargs=search_kwargs
#             )
#             logger.info(f"Retrieving documents for query: {question[:50]}... Filter: {search_kwargs.get('filter', 'none')}")
#             try:
#                 docs = retriever.invoke(question)
#                 if docs is None:
#                     logger.warning("Retriever returned None, defaulting to empty list")
#                     docs = []
#             except AttributeError:
#                 docs = retriever.get_relevant_documents(question)
#                 if docs is None:
#                     logger.warning("Retriever returned None, defaulting to empty list")
#                     docs = []
#             logger.info(f"Retrieved {len(docs)} documents")
#             return docs
#         # Function to process and chunk documents
#         def process_and_chunk(docs):
#             logger.info(f"Processing {len(docs)} documents")
#             if not docs:
#                 logger.warning("No relevant documents found")
#                 return "No relevant documents found. Please provide more specific details or check the vectorstore."
#             valid_texts = []
#             for doc in docs:
#                 if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
#                     valid_texts.append(doc.page_content)
#                 else:
#                     logger.warning(f"Invalid document: {str(doc)}")
#             context = " ".join(valid_texts) if valid_texts else "Limited information available."
#             chunks = text_splitter.split_text(context) if context else [context]
#             logger.info(f"Generated {len(chunks)} chunks")
#             return "\n".join(chunks[:3])

#         # Define prompt
#         prompt = PromptTemplate.from_template(
#             """You are an expert on Indian law assisting lawyers and laymen. Using the provided context, answer concisely in simple language. For statutory questions (e.g., IPC 302), quote the definition and explain in 2-3 simple sentences. For case summaries, provide 3 bullet points (key facts, court, outcome). For court notices, analyze legal implications, suggest 2-3 specific next steps (e.g., hire a lawyer, file a bail application), and list 1-2 opponent arguments with counterpoints, using simple terms. If data is limited, say 'Limited information available' and provide general legal advice.

# Context:
# {context}

# Question: {question}

# Answer:"""
#         )

#         # Initialize LLM
#         llm = ChatOpenAI(
#             model_name="gpt-3.5-turbo",
#             openai_api_key=OPENAI_API_KEY,
#             temperature=0.2,
#             max_tokens=512
#         )
#         logger.info("Initialized LLM: gpt-3.5-turbo")

#         # Create RAG chain
#         chain = (
#             {"context": RunnableLambda(get_relevant_documents) | process_and_chunk,
#              "question": RunnablePassthrough()}
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#         logger.info("RAG chain initialized")
#         return chain, vectorstore
#     except Exception as e:
#         logger.error(f"Failed to initialize RAG chain: {str(e)}")
#         raise

# def process_notice_query(file_path):
#     """Process notice file and integrate with RAG."""
#     try:
#         notice_data = process_notice(file_path)
#         text = notice_data["text"]
#         metadata = notice_data["metadata"]
#         query = f"Analyze the court notice: {text}\nMetadata: {str(metadata)}\nExplain legal implications, suggest next steps, anticipate opponent arguments."
        
#         cached_result = load_from_cache(query)
#         if cached_result:
#             logger.info(f"Retrieved cached result for notice: {query[:50]}...")
#             return cached_result

#         chain, _ = initialize_rag_chain()
#         logger.info(f"Invoking chain with input: {{'question': {query}, 'metadata': {metadata}}}")
#         result = chain.invoke({"question": query, "metadata": metadata})
#         save_to_cache(query, result)
#         logger.info(f"Processed notice query for {file_path}: {result}")
#         return result
#     except Exception as e:
#         logger.error(f"Notice query failed for {file_path}: {str(e)}")
#         raise

# if __name__ == "__main__":
#     chain, vectorstore = initialize_rag_chain()
#     queries = [
#         "Explain IPC 420",
#         "Summarize a CrPC 482 case",
#         "What is the punishment for murder under IPC 302?",
#         "Summarize a case related to IPC 302",
#         "Explain Section 302 IPC in simple terms",
#         "What are the key elements of IPC 302?",
#         "Explain Section 138 NI Act",
#         "Key CrPC 482 judgments",
#         "Describe IPC 498A cases",
#         "Explain Article 21 of the Indian Constitution",
#         {"question": "Analyze the court notice", "file_path": "data/sample_notice.png"}
#     ]
#     for query in queries:
#         try:
#             if isinstance(query, dict) and "file_path" in query:
#                 result = process_notice_query(query["file_path"])
#                 logger.info(f"Query: Analyze notice {query['file_path']}, Answer: {result}")
#                 print(f"Query: Analyze notice {query['file_path']}\nAnswer: {result}\n")
#             else:
#                 cached_result = load_from_cache(query)
#                 if cached_result:
#                     logger.info(f"Query: {query}, Answer: {cached_result}")
#                     print(f"Query: {query}\nAnswer: {cached_result}\n")
#                     continue
#                 logger.info(f"Processing query: {query}")
#                 result = chain.invoke({"question": query, "metadata": None})
#                 save_to_cache(query, result)
#                 logger.info(f"Query: {query}, Answer: {result}")
#                 print(f"Query: {query}\nAnswer: {result}\n")
#             time.sleep(10)
#         except Exception as e:
#             logger.error(f"Query failed: {query}, Error: {str(e)}")


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
    """Extract legal sections (e.g., IPC 420, CrPC 482) from query or notice text."""
    patterns = [
        r'IPC\s*\d+[A-Za-z]?',
        r'Section\s*\d+[A-Za-z]?\s*(?:of\s*(?:Negotiable\s*Instruments\s*Act|NI\s*Act))',
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
        case_number = re.search(r'(CRL|WP|CA|MA)/\d+/\d{4}', text, re.IGNORECASE)
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

    # Define prompt
    prompt_template = PromptTemplate(
        input_variables=["query", "context", "metadata"],
        template="""You are a legal assistant for Indian law. Based on the query and context from Indian Kanoon cases, provide a concise response:
        - For statutory questions, quote the law and explain in 2-3 simple sentences.
        - For case summaries, provide 3 bullet points (key facts, court, outcome).
        - For notices, explain legal implications, 2-3 next steps, 1-2 opponent arguments with counterpoints.
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