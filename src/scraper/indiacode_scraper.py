# src/scraper/indiacode_scraper.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import requests
import fitz  # PyMuPDF
import json
from src.utils.logger import logger

def download_pdf(url, output_path):
    """
    Download a PDF from a URL and save it to output_path.
    Args:
        url (str): URL of the PDF.
        output_path (str): Path to save the PDF.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded PDF to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading PDF from {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text or None if failed.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        doc.close()
        logger.info(f"Extracted text from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def scrape_indiacode_pdf(pdf_url, output_dir="data/raw/indiacode"):
    """
    Scrape an India Code PDF and save the extracted data as JSON.
    Args:
        pdf_url (str): URL of the India Code PDF.
        output_dir (str): Directory to save the JSON output.
    Returns:
        dict: Extracted data or None if failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = pdf_url.split("/")[-1]
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Download the PDF
    if not download_pdf(pdf_url, pdf_path):
        return None

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return None

    # Create data dictionary
    data = {
        "url": pdf_url,
        "title": "Indian Penal Code, 1860",  # Hardcoded for now; can be dynamic later
        "source": "India Code",
        "text": text
    }

    # Save to JSON
    json_filename = pdf_filename.replace(".pdf", ".json")
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved extracted data to {json_path}")

    return data

# Test the scraper
if __name__ == "__main__":
    # pdf_url = "https://www.indiacode.nic.in/bitstream/123456789/1770/3/H1984-3.pdf#search=Indian%20Penal%20Code,%201860"
    pdf_url = "https://www.indiacode.nic.in/bitstream/123456789/4219/1/ipc_1860.pdf"
    result = scrape_indiacode_pdf(pdf_url)
    if result:
        print(json.dumps(result, indent=4))