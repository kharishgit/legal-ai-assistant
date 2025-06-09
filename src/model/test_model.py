# src/model/test_model.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logger import logger

def test_model(model_path="models/inlegalbert-qa"):
    """
    Test the fine-tuned model with sample questions.
    """
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Sample contexts (from your dataset or manually provided)
    context_indiacode = "Section 302. Punishment for murder. Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    context_case = "A Bench of Justices MR Shah and Krishna Murari ruled that reducing a life sentence in a murder case to the period already undergone in prison was contrary to Section 302 of IPC."

    questions = [
        {"question": "What is the punishment under IPC 302?", "context": context_indiacode},
        {"question": "What does this case say about IPC 302?", "context": context_case},
    ]

    for qa in questions:
        result = qa_pipeline(question=qa["question"], context=qa["context"], top_k=1)  # Ensure top_k=1 to get a single answer
        logger.info(f"Question: {qa['question']}")
        # If top_k=1, result is a dict; otherwise, it's a list of dicts
        if isinstance(result, list):
            answer = result[0]["answer"] if result else "No answer found"
        else:
            answer = result["answer"] if result else "No answer found"
        logger.info(f"Answer: {answer}")

if __name__ == "__main__":
    test_model()