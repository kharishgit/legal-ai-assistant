from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def setup_rag():
    """Initialize RAG pipeline."""
    try:
        # Initialize embedding model for ChromaDB
        logger.info("Initializing embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB")
        vectorstore = Chroma(
            collection_name="legal_cases",
            embedding_function=embeddings,
            persist_directory="data/vector_db"
        )
        
        # Initialize InLegalBERT QA model
        logger.info("Loading InLegalBERT model")
        model = AutoModelForQuestionAnswering.from_pretrained("models/inlegalbert-qa")
        tokenizer = AutoTokenizer.from_pretrained("models/inlegalbert-qa")
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        return vectorstore, qa_pipeline
    except Exception as e:
        logger.error(f"Error in setup_rag: {str(e)}")
        raise

def answer_query(query, vectorstore, qa_pipeline, top_k=3):
    """Answer a query using RAG."""
    try:
        logger.info(f"Processing query: {query}")
        # Retrieve relevant documents
        docs = vectorstore.similarity_search(query, k=top_k)
        if not docs:
            logger.warning("No documents retrieved")
            return "No relevant information found."
        
        # Combine documents into context
        context = "\n".join([doc.page_content for doc in docs])
        logger.info(f"Retrieved context: {context[:500]}...")
        
        # Generate answer using InLegalBERT
        result = qa_pipeline(question=query, context=context)
        answer = result["answer"].strip()
        logger.info(f"Answer: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Error in answer_query: {str(e)}")
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    try:
        vectorstore, qa_pipeline = setup_rag()
        queries = [
            "What is the punishment for murder under IPC 302?",
            "Summarize a case related to IPC 302"
        ]
        for query in queries:
            answer = answer_query(query, vectorstore, qa_pipeline)
            print(f"Q: {query}\nA: {answer}\n")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")