# src/scraper/indkanoon_scraper.py
import json
import os
from src.scraper.utils import fetch_page
from src.utils.logger import logger

def scrape_case(url, output_dir="data/raw"):
    """
    Scrape a single IndianKanoon case page and save as JSON.
    Args:
        url (str): URL of the case page.
        output_dir (str): Directory to save JSON output.
    Returns:
        dict: Scraped case data or None if failed.
    """
    # Fetch page
    soup = fetch_page(url)
    if not soup:
        logger.error(f"Failed to scrape case: {url}")
        return None

    # Extract data
    try:
        # Title (e.g., "Maneka Gandhi vs Union Of India")
        title = soup.find("h2", class_="doc_title")
        title = title.text.strip() if title else "Unknown Title"

        # Court and date (e.g., "Supreme Court of India on 25 January, 1978")
        source = soup.find("div", class_="docsource_main")
        court = source.text.split(" on ")[0].strip() if source else "Unknown Court"
        date = source.text.split(" on ")[1].strip() if source and " on " in source.text else "Unknown Date"

        # Case text (main content in <pre> or <div>)
        content = soup.find("pre") or soup.find("div", class_="doc_content")
        content_text = content.text.strip() if content else "No content found"

        # Create data dictionary
        case_data = {
            "url": url,
            "title": title,
            "court": court,
            "date": date,
            "content": content_text
        }

        # Save to JSON
        os.makedirs(output_dir, exist_ok=True)
        case_id = url.split("/")[-2]  # Extract document ID from URL
        output_path = os.path.join(output_dir, f"case_{case_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved case data to {output_path}")

        return case_data
    except Exception as e:
        logger.error(f"Error parsing case {url}: {e}")
        return None

# Test the scraper
if __name__ == "__main__":
    test_url = "https://indiankanoon.org/doc/1786930/"  # Maneka Gandhi case
    case_data = scrape_case(test_url)
    if case_data:
        print(json.dumps(case_data, indent=4))