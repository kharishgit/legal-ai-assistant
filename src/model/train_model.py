# src/model/train_model.py
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def load_model_and_tokenizer(model_name="law-ai/InLegalBERT"):
    """
    Load a pre-trained model and tokenizer.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    if model and tokenizer:
        logger.info("Model and tokenizer loaded successfully")
    else:
        logger.error("Failed to load model and tokenizer")