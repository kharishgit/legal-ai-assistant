# src/model/train_model.py
import json
import torch
import os
import sys
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.logger import logger

def prepare_qa_data(dataset_file="data/processed/combined_dataset.json"):
    """
    Prepare the dataset for question-answering fine-tuning with specific answer spans.
    Returns:
        Dataset: Hugging Face dataset object.
    """
    logger.info(f"Loading dataset from {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_data = []
    for entry in data:
        context = entry.get("text", "")
        source = entry.get("source", "")

        if not context or not isinstance(context, str) or context.strip() == "No content found":
            logger.warning(f"Skipping entry: Invalid or empty context - {context[:50]}... (Source: {source})")
            continue

        # Check for IPC 302 punishment in indiankanoon entries
        if source == "indiankanoon" and "Section 302 in The Indian Penal Code" in context:
            question = "What is the punishment under IPC 302?"
            answer_text = "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
            answer_start = context.find(answer_text)
            if answer_start != -1 and answer_text:
                qa_entry = {
                    "context": context,
                    "question": question,
                    "answers": {"text": [answer_text], "answer_start": [answer_start]}
                }
                qa_data.append(qa_entry)
                logger.debug(f"Added Indian Kanoon QA entry for IPC 302: {qa_entry}")
            else:
                logger.warning(f"Skipping Indian Kanoon entry: Answer not found in context - {context[:50]}... (Source: {source})")

        # Handle case-related questions for both indiankanoon and barandbench
        question = "What does this case say about IPC 302?"
        if source == "indiankanoon" and "Section 302 in The Indian Penal Code" not in context:
            # For case documents, look for sentences with "302" and extract relevant info
            sentences = [s.strip() for s in context.split(".") if "302" in s and s.strip()]
            answer_text = None
            for sentence in sentences:
                if "convicted" in sentence.lower() or "sentence" in sentence.lower():
                    answer_text = sentence
                    break
            if not answer_text and sentences:
                answer_text = sentences[0]
        elif source == "barandbench":
            # For Bar & Bench, look for the sentence with the ruling
            sentences = [s.strip() for s in context.split(".") if "302" in s and s.strip()]
            answer_text = None
            for sentence in sentences:
                if "contrary to Section 302" in sentence:
                    answer_text = sentence
                    break
            if not answer_text and sentences:
                answer_text = sentences[0]
        else:
            continue

        if answer_text and len(answer_text) > 0:
            answer_start = context.find(answer_text)
            if answer_start != -1:
                qa_entry = {
                    "context": context,
                    "question": question,
                    "answers": {"text": [answer_text], "answer_start": [answer_start]}
                }
                qa_data.append(qa_entry)
                logger.debug(f"Added case QA entry: {qa_entry}")
            else:
                logger.warning(f"Skipping case entry: Answer span not found in context - {context[:50]}... (Source: {source})")
        else:
            logger.warning(f"Skipping case entry: No relevant answer found in context - {context[:50]}... (Source: {source})")

    if not qa_data:
        logger.error("No valid QA examples prepared. Check the dataset for issues.")
        raise ValueError("No valid QA examples prepared. Cannot proceed with training.")

    dataset = Dataset.from_list(qa_data)
    logger.info(f"Prepared {len(qa_data)} QA examples")
    return dataset
def tokenize_data(dataset, tokenizer):
    """
    Tokenize the dataset for training with accurate answer span alignment.
    Args:
        dataset (Dataset): QA dataset.
        tokenizer: Hugging Face tokenizer.
    Returns:
        Dataset: Tokenized dataset.
    """
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [c.strip() for c in examples["context"]]
        inputs = tokenizer(
            questions,
            contexts,
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        # Handle overflowing tokens by mapping each tokenized segment to its original example
        sample_mapping = inputs["overflow_to_sample_mapping"] if "overflow_to_sample_mapping" in inputs else list(range(len(questions)))
        
        for i, offset in enumerate(offset_mapping):
            # Map the tokenized segment back to the original example index
            sample_idx = sample_mapping[i]
            
            # Validate answers field for this sample
            if sample_idx >= len(answers) or not answers[sample_idx].get("text") or not answers[sample_idx]["text"] or not isinstance(answers[sample_idx]["text"], list) or len(answers[sample_idx]["text"]) == 0:
                logger.warning(f"Invalid answer for sample {sample_idx} at segment {i}: {answers[sample_idx] if sample_idx < len(answers) else 'None'}. Skipping.")
                start_positions.append(0)
                end_positions.append(0)
                continue

            answer = answers[sample_idx]["text"][0]
            answer_start = answers[sample_idx]["answer_start"][0]
            answer_end = answer_start + len(answer)

            start_pos = 0
            end_pos = 0
            found_start = False
            for idx, (start, end) in enumerate(offset):
                if start <= answer_start < end:
                    start_pos = idx
                    found_start = True
                if found_start and start < answer_end <= end:
                    end_pos = idx
                    break
            if not found_start:
                start_pos = end_pos = 0

            start_positions.append(start_pos)
            end_positions.append(end_pos if end_pos > start_pos else start_pos)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "answers"])
    return tokenized_dataset
def train_model(model_name="law-ai/InLegalBERT", dataset_file="data/processed/combined_dataset.json", output_dir="models/inlegalbert-qa"):
    """
    Fine-tune InLegalBERT for question-answering.
    """
    logger.info(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = prepare_qa_data(dataset_file)
    tokenized_dataset = tokenize_data(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Increase epochs for better learning
        per_device_train_batch_size=4,  # Reduce batch size for stability
        save_steps=200,
        save_total_limit=2,
        logging_dir="logs",
        logging_steps=50,
        learning_rate=3e-5,  # Adjust learning rate
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train_model()