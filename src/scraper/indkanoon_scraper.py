# src/scraper/indkanoon_scraper.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from src.scraper.utils import fetch_page, get_case_links
from src.utils.logger import logger

def scrape_case(url, output_dir="data/raw"):
    soup = fetch_page(url, use_playwright=True)  # Use Playwright for case pages
    if not soup:
        logger.error(f"Failed to scrape case: {url}")
        return None

    try:
        title = soup.find("h2", class_="doc_title")
        title = title.text.strip() if title else "Unknown Title"

        source = soup.find("div", class_="docsource_main")
        court = source.text.split(" on ")[0].strip() if source else "Unknown Court"
        date = source.text.split(" on ")[1].strip() if source and " on " in source.text else "Unknown Date"

        content = soup.find("pre") or soup.find("div", class_="doc_content")
        content_text = content.text.strip() if content else "No content found"

        case_data = {
            "url": url,
            "title": title,
            "court": court,
            "date": date,
            "content": content_text
        }

        os.makedirs(output_dir, exist_ok=True)
        case_id = url.split("/")[-2]
        output_path = os.path.join(output_dir, f"case_{case_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved case data to {output_path}")

        return case_data
    except Exception as e:
        logger.error(f"Error parsing case {url}: {e}")
        return None

def scrape_search_page(search_url, max_cases=10, output_dir="data/raw"):
    case_links = get_case_links(search_url, max_cases)
    if not case_links:
        logger.error(f"No case links found for {search_url}. Check debug_search.html for HTML structure.")
        return []

    cases_data = []
    for url in case_links:
        logger.info(f"Scraping case: {url}")
        case_data = scrape_case(url, output_dir)
        if case_data:
            cases_data.append(case_data)
    return cases_data

if __name__ == "__main__":
    search_url = "https://indiankanoon.org/search/?formInput=ipc%20302"
    cases = scrape_search_page(search_url, max_cases=10)
    if not cases:
        print("No cases scraped. Check app.log and debug_search.html for details.")
    else:
        for case in cases:
            print(json.dumps(case, indent=4))