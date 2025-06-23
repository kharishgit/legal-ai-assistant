from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error in chunk_text: {str(e)}")
        raise