import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import os
import re
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from src.utils.logger import logger
import pickle
import hashlib

# Cache directory for OCR results
CACHE_DIR = "data/cache/ocr"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def load_from_cache(text):
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(text)}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(text, result):
    cache_file = os.path.join(CACHE_DIR, f"{get_cache_key(text)}.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

def extract_metadata(text):
    """Extract metadata from OCR text using regex."""
    metadata = {}
    try:
        # Extract case number
        case_pattern = r"Case No[\.:]?\s*([A-Z0-9/]+)"
        case_match = re.search(case_pattern, text, re.IGNORECASE)
        if case_match:
            metadata["case_number"] = case_match.group(1)

        # Extract sections (e.g., IPC 302, NI Act 138)
        section_pattern = r"(?:Section|Sec\.)\s*(\d+[A-Za-z]?(?:\s*IPC|\s*NI\s*Act)?)"
        sections = re.findall(section_pattern, text, re.IGNORECASE)
        if sections:
            metadata["sections"] = sections

        # Extract court (improved to avoid extra text)
        court_pattern = r"(?:High Court|Supreme Court|Court)\s*(?:of|at)?\s*([A-Za-z\s]+?)(?:\n|$)"
        court_match = re.search(court_pattern, text, re.IGNORECASE)
        if court_match:
            metadata["court"] = court_match.group(1).strip()

        # Extract date
        date_pattern = r"Issued on[\.:]?\s*(\d{1,2}-\d{1,2}-\d{4})"
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            metadata["date"] = date_match.group(1)

        logger.info(f"Extracted metadata: {metadata}")
        return metadata
    except Exception as e:
        logger.error(f"Metadata extraction failed: {str(e)}")
        return metadata

def process_image(image_path):
    """Process image file with OCR and extract metadata."""
    try:
        cached_result = load_from_cache(image_path)
        if cached_result:
            logger.info(f"Retrieved cached OCR result for {image_path}")
            return cached_result

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        metadata = extract_metadata(text)
        result = {"text": text, "metadata": metadata}
        save_to_cache(image_path, result)
        logger.info(f"Processed image {image_path}: {result}")
        return result
    except Exception as e:
        logger.error(f"Image processing failed for {image_path}: {str(e)}")
        raise

def process_pdf(pdf_path):
    """Process PDF file with OCR and extract metadata."""
    try:
        cached_result = load_from_cache(pdf_path)
        if cached_result:
            logger.info(f"Retrieved cached OCR result for {pdf_path}")
            return cached_result

        images = convert_from_path(pdf_path)
        texts = []
        metadata = {}
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            texts.append(text)
            page_metadata = extract_metadata(text)
            for key, value in page_metadata.items():
                if key == "sections":
                    metadata.setdefault(key, []).extend(value)
                else:
                    metadata[key] = value
        result = {"text": "\n".join(texts), "metadata": metadata}
        save_to_cache(pdf_path, result)
        logger.info(f"Processed PDF {pdf_path}: {result}")
        return result
    except Exception as e:
        logger.error(f"PDF processing failed for {pdf_path}: {str(e)}")
        raise

def process_notice(file_path):
    """Process image or PDF notice and return text and metadata."""
    try:
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return process_image(file_path)
        elif file_path.lower().endswith('.pdf'):
            return process_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    except Exception as e:
        logger.error(f"Notice processing failed for {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    sample_notice = "data/sample_notice.png"
    result = process_notice(sample_notice)
    print(f"Notice Text: {result['text']}")
    print(f"Metadata: {result['metadata']}")