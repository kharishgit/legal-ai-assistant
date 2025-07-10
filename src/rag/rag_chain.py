import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ocr.ocr_processor import process_notice
from src.utils.logger import logger
import pickle
import hashlib
import sys
import json

# Add parent directory to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Disable HuggingFace tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Simple caching mechanism
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

def load_from_cache(query):
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(query, result):
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(query)}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

def initialize_rag_chain():
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        # Initialize vectorstore
        vectorstore = Chroma(
            persist_directory="data/vector_db",
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        )
        logger.info("Initialized ChromaDB for RAG")

        # Function to configure retriever and fetch documents
        def get_relevant_documents(input_data):
            question = input_data["question"]
            metadata = input_data.get("metadata", {})
            search_kwargs = {"k": 3, "fetch_k": 10}
            if metadata and "sections" in metadata:
                search_kwargs["filter"] = {"sections": {"$in": metadata["sections"]}}
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
            logger.info(f"Retrieving documents for query: {question[:50]}...")
            try:
                docs = retriever.invoke(question)
                if docs is None:
                    logger.warning("Retriever returned None, defaulting to empty list")
                    docs = []
            except AttributeError:
                docs = retriever.get_relevant_documents(question)
                if docs is None:
                    logger.warning("Retriever returned None, defaulting to empty list")
                    docs = []
            logger.info(f"Retrieved {len(docs)} documents")
            return docs

        # Function to process and chunk documents
        def process_and_chunk(docs):
            logger.info(f"Processing {len(docs)} documents")
            if not docs:
                logger.warning("No relevant documents found")
                return "No relevant documents found. Please provide more specific details or check the vectorstore."
            valid_texts = []
            for doc in docs:
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                    valid_texts.append(doc.page_content)
                else:
                    logger.warning(f"Invalid document: {str(doc)}")
            context = " ".join(valid_texts) if valid_texts else "Limited information available."
            chunks = text_splitter.split_text(context) if context else [context]
            logger.info(f"Generated {len(chunks)} chunks")
            return "\n".join(chunks[:3])

        # Define prompt
        prompt = PromptTemplate.from_template(
            """You are an expert on Indian law assisting lawyers and laymen. Using the provided context, answer concisely in simple language. For statutory questions (e.g., IPC 302), quote the definition and explain in 2-3 simple sentences. For case summaries, provide 3 bullet points (key facts, court, outcome). For court notices, analyze legal implications, suggest 2-3 specific next steps (e.g., hire a lawyer, file a bail application), and list 1-2 opponent arguments with counterpoints, using simple terms. If data is limited, say 'Limited information available' and provide general legal advice.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=512
        )
        logger.info("Initialized LLM: gpt-3.5-turbo")

        # Create RAG chain
        chain = (
            {"context": RunnableLambda(get_relevant_documents) | process_and_chunk,
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized")
        return chain, vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise

def process_notice_query(file_path):
    """Process notice file and integrate with RAG."""
    try:
        notice_data = process_notice(file_path)
        text = notice_data["text"]
        metadata = notice_data["metadata"]
        query = f"Analyze the court notice: {text}\nMetadata: {str(metadata)}\nExplain legal implications, suggest next steps, anticipate opponent arguments."
        
        cached_result = load_from_cache(query)
        if cached_result:
            logger.info(f"Retrieved cached result for notice: {query[:50]}...")
            return cached_result

        chain, _ = initialize_rag_chain()
        logger.info(f"Invoking chain with input: {{'question': {query}, 'metadata': {metadata}}}")
        result = chain.invoke({"question": query, "metadata": metadata})
        save_to_cache(query, result)
        logger.info(f"Processed notice query for {file_path}: {result}")
        return result
    except Exception as e:
        logger.error(f"Notice query failed for {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    chain, vectorstore = initialize_rag_chain()
    queries = [
        "What is the punishment for murder under IPC 302?",
        "Summarize a case related to IPC 302",
        "Explain Section 302 IPC in simple terms",
        "What are the key elements of IPC 302?",
        "Explain Section 138 NI Act",
        "Key CrPC 482 judgments",
        "Describe IPC 498A cases",
        "Explain Article 21 of the Indian Constitution",
        {"question": "Analyze the court notice", "file_path": "data/sample_notice.png"}
    ]
    for query in queries:
        try:
            if isinstance(query, dict) and "file_path" in query:
                result = process_notice_query(query["file_path"])
                logger.info(f"Query: Analyze notice {query['file_path']}, Answer: {result}")
                print(f"Query: Analyze notice {query['file_path']}\nAnswer: {result}\n")
            else:
                cached_result = load_from_cache(query)
                if cached_result:
                    logger.info(f"Query: {query}, Answer: {cached_result}")
                    print(f"Query: {query}\nAnswer: {cached_result}\n")
                    continue
                logger.info(f"Processing query: {query}")
                result = chain.invoke({"question": query, "metadata": None})
                save_to_cache(query, result)
                logger.info(f"Query: {query}, Answer: {result}")
                print(f"Query: {query}\nAnswer: {result}\n")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Query failed: {query}, Error: {str(e)}")