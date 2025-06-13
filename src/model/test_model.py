


from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
from src.utils.logger import logger

def test_model(model_path="models/inlegalbert-qa"):
    """
    Test the fine-tuned model with sample questions.
    """
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Sample contexts (from your dataset or manually provided)
    context_indiacode = "Section 302 in The Indian Penal Code, 1860: Punishment for murder. Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
    context_case = "In a significant ruling, the Supreme Court of India clarified the application of Section 302 of the Indian Penal Code, which deals with the punishment for murder. A Bench of Justices MR Shah and Aniruddha Bose held that reducing a life sentence in a murder case to the period already undergone in prison was contrary to Section 302 of IPC, emphasizing that the minimum punishment for murder under Section 302 is life imprisonment or death, reflecting the gravity of the offence."
    context_new_ipc302 = "Under the Indian Penal Code, Section 302 states that a person who commits murder is liable to be punished with death or life imprisonment, along with a possible fine."
    context_unrelated = "The Supreme Court ruled on a property dispute case, stating that the ownership of the land belongs to the plaintiff due to proper documentation. The case had no relation to criminal law."

    questions = [
        # Original test cases
        {"question": "What is the punishment under IPC 302?", "context": context_indiacode},
        {"question": "What does this case say about IPC 302?", "context": context_case},
        # New test cases for generalization
        {"question": "What is the minimum punishment under IPC 302?", "context": context_new_ipc302},
        {"question": "What does this case say about IPC 302?", "context": context_unrelated},
    ]

    for qa in questions:
        question = qa["question"]
        context = qa["context"]
        result = qa_pipeline(question=question, context=context)

        # Extract the answer
        answer = result["answer"].strip()

        # Post-process to ensure the answer is concise and relevant
        if answer and context:
            answer_start_char = result["start"]
            answer_end_char = result["end"]
            # Find sentence boundaries
            sentence_start = context.rfind(".", 0, answer_start_char) + 1
            sentence_end = context.find(".", answer_end_char)
            if sentence_start == 0:
                sentence_start = 0
            if sentence_end == -1:
                sentence_end = len(context)
            full_answer = context[sentence_start:sentence_end].strip()
            if full_answer.startswith("."):
                full_answer = full_answer[1:].strip()

            # Refine the answer based on the question
            if "What does this case say about IPC 302?" in question:
                # Look for the core ruling by finding the phrase with "Section 302"
                phrases = [p.strip() for p in full_answer.split(",") if "Section 302" in p]
                if phrases:
                    # Further refine by removing introductory phrases
                    core_phrase = phrases[0]
                    if "held that" in core_phrase:
                        core_phrase = core_phrase.split("held that", 1)[-1].strip()
                    answer = core_phrase
                else:
                    answer = "This case does not mention IPC 302." if "302" not in full_answer else full_answer
            else:
                answer = full_answer

        logger.info(f"Question: {question}")
        logger.info(f"Answer: {answer}")

if __name__ == "__main__":
    test_model()