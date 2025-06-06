# src/data/combine_dataset.py
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import logger

def combine_dataset(input_files, output_file="data/processed/combined_dataset.json"):
    """
    Combine multiple processed JSON files into a single dataset.
    Args:
        input_files (list): List of input JSON file paths.
        output_file (str): Path to save the combined dataset.
    Returns:
        list: Combined dataset entries.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_data = []

    for filepath in input_files:
        logger.info(f"Combining data from {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            combined_data.extend(data)
        except Exception as e:
            logger.error(f"Error combining {filepath}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved combined dataset to {output_file}")
    return combined_data

if __name__ == "__main__":
    input_files = [
        "data/processed/indiacode_processed.json",
        "data/processed/barandbench_processed.json",
        "data/processed/indiankanoon_processed.json"
    ]
    combined_data = combine_dataset(input_files)
    logger.info(f"Combined dataset contains {len(combined_data)} entries")