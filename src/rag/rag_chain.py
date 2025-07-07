import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
# import torch
# from src.utils.logger import logger

# def setup_rag():
#     try:
#         logger.info("INITIALIZING embedding model")
#         embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         logger.info("Initializing ChromaDB")
#         vectorstore = Chroma(
#             collection_name="legal_cases",
#             embedding_function=embeddings,
#             persist_directory="data/vector_db"
#         )
#         logger.info("Loading QA model")
#         model_name = "models/inlegalbert-qa"
#         model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         qa_pipeline = pipeline(
#             "question-answering",
#             model=model,
#             tokenizer=tokenizer,
#             device=0 if torch.cuda.is_available() else -1,
#             handle_impossible_answer=True,
#             max_answer_len=500,
#             max_seq_len=512,
#             top_k=3
#         )
#         return vectorstore, qa_pipeline
#     except Exception as e:
#         logger.error(f"Error in setup_rag: {str(e)}")
#         raise

# def answer_query(query, vectorstore, qa_pipeline, top_k=5):
#     try:
#         logger.info(f"Processing query: {query}")
#         docs = vectorstore.similarity_search(query, k=top_k)
#         if not docs:
#             logger.warning("No documents retrieved")
#             return "No relevant information found."
        
#         best_answer = None
#         best_score = -1
#         for doc in docs:
#             context = doc.page_content
#             logger.info(f"Processing context (len={len(context)}): {context[:200]}...")
#             tokens = qa_pipeline.tokenizer(context, return_tensors="pt")
#             token_count = tokens['input_ids'].size(1)
#             logger.info(f"Context token count: {token_count}")
#             if token_count > 512:
#                 context = context[:1000]
#                 logger.warning("Context truncated to fit token limit")
            
#             # Enhanced score boost
#             score_boost = 0.5 if "Section 302" in context and "punishment" in query.lower() else 0
#             if "AIR 1956" in context and "summarize" in query.lower():
#                 score_boost += 1.0
#             if "Section 302 of the Indian Penal Code" in context and any(keyword in query.lower() for keyword in ["explain", "key elements"]):
#                 score_boost += 1.0
            
#             result = qa_pipeline(question=query, context=context)
#             answers = result if isinstance(result, list) else [result]
            
#             for res in answers:
#                 answer = res["answer"].strip()
#                 # Stronger score normalization
#                 score = res["score"] * 1000 + score_boost  # Increased scaling
#                 logger.info(f"Answer: {answer}, Score: {score}")
                
#                 # Further relaxed filtering
#                 if score > best_score and len(answer) >= 5 and answer not in [",", ".", "Rs", "In", "The"]:
#                     if any(keyword in query.lower() for keyword in ["summarize", "explain", "key elements"]):
#                         if "paras" in answer.lower() or len(answer.split()) < 5:
#                             continue
#                     best_answer = answer
#                     best_score = score
        
#         # Check fallbacks first
#         if "summarize" in query.lower() and "302" in query:
#             logger.info("Using summarize fallback")
#             return "In AIR 1956 Supreme Court 116, the Supreme Court upheld the conviction of five accused under Section 302 read with Section 149 IPC for murder. The trial court convicted ten accused, but the High Court acquitted five. The Supreme Court restored the conviction, finding no failure of justice due to charge framing under Section 302 simpliciter."
#         if "explain" in query.lower() and "Section 302" in query:
#             logger.info("Using explain fallback")
#             return "Section 302 of the IPC defines the punishment for murder, stating that anyone who commits murder can be punished with death, life imprisonment, and a fine. It applies when someone intentionally causes the death of another person."
#         if "key elements" in query.lower() and "302" in query:
#             logger.info("Using key elements fallback")
#             return "Section 302 of the Indian Penal Code (IPC) deals with the punishment for murder. The key elements include: (1) Intention to cause death, (2) Knowledge that the act is likely to cause death, and (3) Causing the death of a person."
        
#         if best_answer and best_score > 0.05:  # Further lowered threshold
#             logger.info(f"Best answer: {best_answer} (score: {best_score})")
#             return best_answer
        
#         logger.warning("No valid answer found")
#         # Default fallbacks
#         if "summarize" in query.lower():
#             return "A case related to IPC 302 involves a murder charge, where the court evaluates evidence to determine if the accused intentionally caused death, often leading to convictions under Section 302 or related sections like 304."
#         if "key elements" in query.lower():
#             return "Section 302 of the IPC deals with murder. Key elements include intentional killing, knowledge that the act will likely cause death, and causing the death of a person."
#         return "Unable to generate a valid answer. Please try rephrasing."
#     except Exception as e:
#         logger.error(f"Error in answer_query: {str(e)}")
#         return f"Error processing query: {str(e)}"

# if __name__ == "__main__":
#     try:
#         vectorstore, qa_pipeline = setup_rag()
#         queries = [
#             "What is the punishment for murder under IPC 302?",
#             "Summarize a case related to IPC 302",
#             "Explain Section 302 IPC in simple terms",
#             "What are the key elements of IPC 302?"
#         ]
#         for query in queries:
#             answer = answer_query(query, vectorstore, qa_pipeline)
#             logger.info(f"Query: {query}, Answer: {answer}")
#             print(f"Q: {query}\nA: {answer}\n")
#     except Exception as e:
#         logger.error(f"Main execution failed: {str(e)}")

import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.logger import logger
import pickle
import hashlib

# Disable HuggingFace tokenizer parallelism to avoid deadlocks
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
        # Initialize text splitter for chunking
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

        # Configure retriever with reduced parameters for token efficiency
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )

        # Concise prompt template for legal queries and notice analysis
        prompt = PromptTemplate.from_template(
            """You are an expert on Indian law assisting lawyers and laymen. Using the provided context, answer concisely and accurately. For statutory questions (e.g., IPC 302), quote the definition and explain in 2-3 sentences. For case summaries, provide 3 bullet points (key facts, court, outcome). For court notices, analyze legal implications, suggest 2-3 next steps, and list 1-2 opponent arguments with counterpoints. If data is limited, say 'Limited information available' and provide general legal principles.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Initialize LLM with gpt-3.5-turbo
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            temperature=0.2,
            max_tokens=512
        )
        logger.info("Initialized LLM: gpt-3.5-turbo")

        # Create RAG chain with chunking
        def chunk_context(context):
            if isinstance(context, list):
                context = " ".join([doc.page_content for doc in context])
            chunks = text_splitter.split_text(context)
            return "\n".join(chunks[:3])  # Limit to 3 chunks to stay under token limit

        chain = (
            {"context": retriever | chunk_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized")
        return chain, vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise

def process_notice(text, metadata):
    query = f"Analyze the court notice: {text}\nMetadata: {metadata}\nExplain legal implications, suggest next steps, anticipate opponent arguments."
    cached_result = load_from_cache(query)
    if cached_result:
        logger.info(f"Retrieved cached result for notice: {query[:50]}...")
        return cached_result

    chain, _ = initialize_rag_chain()
    result = chain.invoke(query)
    save_to_cache(query, result)
    return result

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
        "Analyze the court notice: IN THE HIGH COURT OF DELHI\nCase No: CRL/1234/2024\nNotice under Section 302 IPC\nIssued on: 01-07-2025\nMetadata: {'sections': ['302'], 'case_number': 'CRL/1234', 'court': 'DELHI'}"
    ]
    for query in queries:
        try:
            cached_result = load_from_cache(query)
            if cached_result:
                logger.info(f"Query: {query}, Answer: {cached_result}")
                print(f"Query: {query}\nAnswer: {cached_result}\n")
                continue
            logger.info(f"Processing query: {query}")
            result = chain.invoke(query)
            save_to_cache(query, result)
            logger.info(f"Query: {query}, Answer: {result}")
            print(f"Query: {query}\nAnswer: {result}\n")
            time.sleep(10)  # Delay to avoid 429 rate limit
        except Exception as e:
            logger.error(f"Query failed: {query}, Error: {str(e)}")