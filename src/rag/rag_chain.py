import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your actual key

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
            if "sections" in metadata:
                search_kwargs["filter"] = {"sections": {"$in": metadata["sections"]}}
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs=search_kwargs
            )
            # Use invoke for modern LangChain compatibility, fallback to get_relevant_documents
            try:
                docs = retriever.invoke(question)
            except AttributeError:
                docs = retriever.get_relevant_documents(question)
            return docs

        # Function to process and chunk documents
        def process_and_chunk(docs):
            if not docs:
                return "No relevant documents found."
            valid_texts = []
            for doc in docs:
                if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                    valid_texts.append(doc.page_content)
            context = " ".join(valid_texts) if valid_texts else "Limited information available."
            chunks = text_splitter.split_text(context) if context else [context]
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