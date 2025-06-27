
# src/scraper/preprocessor.py
import json
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def clean_text(text):
    """
    Clean text by removing noise (extra whitespace, special characters, etc.).
    Args:
        text (str): Raw text to clean.
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^\w\s.,;:-]', '', text)
    # Remove common PDF artifacts (e.g., page numbers, headers)
    text = re.sub(r'Page \d+', '', text)
    text = re.sub(r'Indian Penal Code, 1860', '', text)  # Remove repetitive headers
    return text

def is_ipc302_relevant(text):
    """
    Check if a text is relevant to IPC 302.
    Args:
        text (str): Text to check (e.g., article title or content).
    Returns:
        bool: True if relevant to IPC 302, False otherwise.
    """
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    # Look for keywords like "IPC 302", "Section 302", or "murder" (context-specific)
    keywords = ["ipc 302", "section 302", "murder under 302", "murder", "homicide"]
    return any(keyword in text_lower for keyword in keywords)

def preprocess_indiacode_data(input_dir="data/raw/indiacode", output_file="data/processed/indiacode_processed.json"):
    """
    Preprocess India Code JSON files, filtering for IPC 302 relevance.
    Args:
        input_dir (str): Directory containing raw India Code JSON files.
        output_file (str): Path to save the processed JSON file.
    Returns:
        list: List of processed data dictionaries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            logger.info(f"Processing India Code file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Clean the extracted text
                cleaned_text = clean_text(data.get("text", ""))
                # Filter for IPC 302 relevance
                if cleaned_text and is_ipc302_relevant(cleaned_text):
                    processed_entry = {
                        "source": "indiacode",
                        "text": cleaned_text,
                        "metadata": {
                            "filename": filename,
                            "original_url": data.get("url", "")
                        }
                    }
                    processed_data.append(processed_entry)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved processed India Code data to {output_file}")
    return processed_data

def preprocess_barandbench_data(input_dir="data/raw/barandbench", output_file="data/processed/barandbench_processed.json"):
    """
    Preprocess Bar & Bench JSON files, filtering for IPC 302 relevance.
    Args:
        input_dir (str): Directory containing raw Bar & Bench JSON files.
        output_file (str): Path to save the processed JSON file.
    Returns:
        list: List of processed data dictionaries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            logger.info(f"Processing Bar & Bench file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Combine title and content for relevance check
                title = data.get("title", "")
                content = data.get("content", "")
                combined_text = f"{title} {content}"
                # Filter for IPC 302 relevance
                if is_ipc302_relevant(combined_text):
                    # Clean the content
                    cleaned_content = clean_text(content)
                    if cleaned_content:
                        processed_entry = {
                            "source": "barandbench",
                            "text": cleaned_content,
                            "metadata": {
                                "title": title,
                                "date": data.get("date", ""),
                                "url": data.get("url", "")
                            }
                        }
                        processed_data.append(processed_entry)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved processed Bar & Bench data to {output_file}")
    return processed_data

def preprocess_indiankanoon_data(input_dir="data/raw/indkanoon", output_file="data/processed/indiankanoon_processed.json"):
    """
    Preprocess Indian Kanoon JSON files, filtering for IPC 302 relevance.
    Args:
        input_dir (str): Directory containing raw Indian Kanoon JSON files.
        output_file (str): Path to save the processed JSON file.
    Returns:
        list: List of processed data dictionaries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            logger.info(f"Processing Indian Kanoon file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Combine title and content for relevance check
                title = data.get("title", "")
                content = data.get("content", "")
                combined_text = f"{title} {content}"
                # Filter for IPC 302 relevance
                if is_ipc302_relevant(combined_text):
                    # Clean the content
                    cleaned_content = clean_text(content)
                    if cleaned_content:
                        processed_entry = {
                            "source": "indiankanoon",
                            "text": cleaned_content,
                            "metadata": {
                                "title": title,
                                "date": data.get("date", ""),
                                "url": data.get("url", "")
                            }
                        }
                        processed_data.append(processed_entry)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved processed Indian Kanoon data to {output_file}")
    return processed_data

def preprocess_newsapi_data(input_dir="data/raw/newsapi", output_file="data/processed/newsapi_processed.json"):
    """
    Preprocess News API JSON files, filtering for IPC 302 relevance.
    Args:
        input_dir (str): Directory containing raw News API JSON files.
        output_file (str): Path to save the processed JSON file.
    Returns:
        list: List of processed data dictionaries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_data = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            logger.info(f"Processing News API file: {filepath}")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Combine title and content for relevance check
                title = data.get("title", "")
                content = data.get("content", "") or data.get("description", "")
                combined_text = f"{title} {content}"
                # Filter for IPC 302 relevance
                if is_ipc302_relevant(combined_text):
                    # Clean the content
                    cleaned_content = clean_text(content)
                    if cleaned_content:
                        processed_entry = {
                            "source": "newsapi",
                            "text": cleaned_content,
                            "metadata": {
                                "title": title,
                                "date": data.get("date", "") or data.get("publishedAt", ""),
                                "url": data.get("url", "")
                            }
                        }
                        processed_data.append(processed_entry)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")

    # Save processed data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved processed News API data to {output_file}")
    return processed_data

if __name__ == "__main__":
    # Preprocess all datasets
    indiacode_data = preprocess_indiacode_data()
    barandbench_data = preprocess_barandbench_data()
    indiankanoon_data = preprocess_indiankanoon_data()
    newsapi_data = preprocess_newsapi_data()
    logger.info(f"Processed {len(indiacode_data)} India Code entries")
    logger.info(f"Processed {len(barandbench_data)} Bar & Bench articles (after IPC 302 filtering)")
    logger.info(f"Processed {len(indiankanoon_data)} Indian Kanoon cases (after IPC 302 filtering)")
    logger.info(f"Processed {len(newsapi_data)} News API articles (after IPC 302 filtering)")