# src/scraper/preprocessor.py
import json
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def clean_text(text):
    """
    Clean a piece of text by removing extra whitespace, newlines, and special characters.
    Args:
        text (str): Text to clean.
    Returns:
        str: Cleaned text.
    """
    if not text or text == "No content found":
        return text
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s.,;!?()-]', '', text)
    return text

def preprocess_case(case_data):
    """
    Preprocess a single case dictionary.
    Args:
        case_data (dict): Case data dictionary.
    Returns:
        dict: Preprocessed case data.
    """
    try:
        # Clean the title, court, date, and content
        case_data["title"] = clean_text(case_data.get("title", ""))
        case_data["court"] = clean_text(case_data.get("court", ""))
        case_data["date"] = clean_text(case_data.get("date", ""))
        case_data["content"] = clean_text(case_data.get("content", ""))
        return case_data
    except Exception as e:
        logger.error(f"Error preprocessing case {case_data.get('url', 'unknown')}: {e}")
        return None

def preprocess_cases(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Preprocess all JSON files in raw_dir and save to processed_dir.
    Args:
        raw_dir (str): Directory with raw JSON files.
        processed_dir (str): Directory to save processed JSON files.
    Returns:
        list: List of preprocessed case data dictionaries.
    """
    os.makedirs(processed_dir, exist_ok=True)
    processed_cases = []

    for filename in os.listdir(raw_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(raw_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    case_data = json.load(f)
                logger.info(f"Preprocessing case: {case_data.get('url', 'unknown')}")
                cleaned_case = preprocess_case(case_data)
                if cleaned_case:
                    # Save cleaned data
                    output_path = os.path.join(processed_dir, filename)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(cleaned_case, f, ensure_ascii=False, indent=4)
                    logger.info(f"Saved preprocessed case to {output_path}")
                    processed_cases.append(cleaned_case)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")

    return processed_cases

# Test the preprocessor
if __name__ == "__main__":
    processed_cases = preprocess_cases()
    for case in processed_cases:
        print(json.dumps(case, indent=4))