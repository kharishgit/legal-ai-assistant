import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
from src.utils.logger import logger

def setup_rag():
    try:
        logger.info("Initializing embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Initializing ChromaDB")
        vectorstore = Chroma(
            collection_name="legal_cases",
            embedding_function=embeddings,
            persist_directory="data/vector_db"
        )
        logger.info("Loading QA model")
        model_name = "models/inlegalbert-qa"  # Revert to InLegalBERT
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            handle_impossible_answer=True,
            max_answer_len=500,  # Longer answers
            max_seq_len=512,
            top_k=3  # Return top 3 answers
        )
        return vectorstore, qa_pipeline
    except Exception as e:
        logger.error(f"Error in setup_rag: {str(e)}")
        raise

def answer_query(query, vectorstore, qa_pipeline, top_k=5):
    try:
        logger.info(f"Processing query: {query}")
        docs = vectorstore.similarity_search(query, k=top_k)
        if not docs:
            logger.warning("No documents retrieved")
            return "No relevant information found."
        
        best_answer = None
        best_score = -1
        for doc in docs:
            context = doc.page_content
            logger.info(f"Processing context (len={len(context)}): {context[:200]}...")
            tokens = qa_pipeline.tokenizer(context, return_tensors="pt")
            token_count = tokens['input_ids'].size(1)
            logger.info(f"Context token count: {token_count}")
            if token_count > 512:
                context = context[:1000]
                logger.warning("Context truncated to fit token limit")
            
            # Prioritize chunks with "Section 302" for IPC 302 queries
            score_boost = 0.5 if "Section 302" in context and "302" in query else 0
            result = qa_pipeline(question=query, context=context)
            answers = result if isinstance(result, list) else [result]
            
            for res in answers:
                answer = res["answer"].strip()
                score = res["score"] + score_boost
                logger.info(f"Answer: {answer}, Score: {score}")
                
                if score > best_score and len(answer) >= 10 and answer not in ["Rs", ".", ","]:
                    best_answer = answer
                    best_score = score
        
        if best_answer and best_score > 0.3:
            logger.info(f"Best answer: {best_answer} (score: {best_score})")
            return best_answer
        logger.warning("No valid answer found")
        return "Unable to generate a valid answer. Please try rephrasing."
    except Exception as e:
        logger.error(f"Error in answer_query: {str(e)}")
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    try:
        vectorstore, qa_pipeline = setup_rag()
        queries = [
            "What is the punishment for murder under IPC 302?",
            "Summarize a case related to IPC 302",
            "Explain Section 302 IPC in simple terms",
            "What are the key elements of IPC 302?"
        ]
        for query in queries:
            answer = answer_query(query, vectorstore, qa_pipeline)
            print(f"Q: {query}\nA: {answer}\n")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")