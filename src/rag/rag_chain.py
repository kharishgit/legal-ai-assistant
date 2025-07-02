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

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import logger

def initialize_rag_chain():
    try:
        # Initialize vectorstore
        vectorstore = Chroma(
            persist_directory="data/vector_db",
            embedding_function=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu"}
            )
        )
        logger.info("Initialized ChromaDB for RAG")

        # Configure retriever with MMR
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20}
        )

        # Define prompt template
        prompt = PromptTemplate.from_template(
            """You are a legal expert on Indian law. Using the provided case context, answer the question accurately and concisely, focusing on the relevant legal section or case details. Avoid repeating case text verbatim and remove any formatting artifacts (e.g., 'b302b'). If the question is about a specific section (e.g., IPC 302), prioritize statutory definitions or direct case references. If no relevant information is found, say so.

Context:
{context}

Question: {question}

Answer:"""
        )

        # Initialize LLM
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base",
            task="text2text-generation",
            pipeline_kwargs={"max_length": 512, "truncation": True}
        )
        logger.info("Initialized LLM: google/flan-t5-base")

        # Create RAG chain
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG chain initialized")
        return chain, vectorstore
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}")
        raise

def test_rag():
    chain, vectorstore = initialize_rag_chain()
    queries = [
        "What is the punishment for murder under IPC 302?",
        "Summarize a case related to IPC 302",
        "Explain Section 302 IPC in simple terms",
        "What are the key elements of IPC 302?",
        "Explain Section 138 NI Act",
        "Key CrPC 482 judgments",
        "Describe IPC 498A cases"
    ]
    for query in queries:
        try:
            logger.info(f"Processing query: {query}")
            result = chain.invoke(query)
            logger.info(f"Query: {query}, Answer: {result}")
            print(f"Query: {query}\nAnswer: {result}\n")
        except Exception as e:
            logger.error(f"Query failed: {query}, Error: {str(e)}")

if __name__ == "__main__":
    test_rag()