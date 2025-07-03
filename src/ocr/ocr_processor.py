import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pytesseract
from PIL import Image
import re
from src.utils.logger import logger

def extract_notice_text(image_path):
    try:
        # Extract text from image
        text = pytesseract.image_to_string(Image.open(image_path))
        logger.info(f"Extracted text from {image_path}: {text[:100]}...")

        # Parse for legal sections (e.g., IPC 302, CrPC 482)
        sections = re.findall(r"(?:IPC|CrPC|NI Act|Section)\s*(\d+[A-Za-z]?)", text, re.IGNORECASE)
        case_number = re.search(r"Case\s*No\.?\s*[:\-]?\s*(\w+/\d+)", text, re.IGNORECASE)
        court = re.search(r"(?:Supreme Court|High Court|District Court)\s*(?:of\s*([\w\s]+))?", text, re.IGNORECASE)

        metadata = {
            "sections": sections if sections else [],
            "case_number": case_number.group(1) if case_number else "Unknown",
            "court": court.group(1) if court else "Unknown"
        }
        logger.info(f"Parsed metadata: {metadata}")
        return text, metadata
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Test with a sample notice image
    sample_image = "sample_notice.png"
    text, metadata = extract_notice_text(sample_image)
    if text:
        print(f"Extracted Text: {text}\nMetadata: {metadata}")