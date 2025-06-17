import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import re
from bs4 import BeautifulSoup
from src.utils.logger import logger
from src.scraper.utils import fetch_page, get_case_links

# def scrape_case(url):
#     """
#     Scrape a single case page from IndianKanoon.
#     Args:
#         url (str): URL of the case page.
#     Returns:
#         dict: Scraped case data or None if failed.
#     """
#     soup = fetch_page(url, use_playwright=True)
#     if not soup:
#         logger.error(f"Failed to scrape case: {url}")
#         return None

#     # Save the HTML of the case page for debugging
#     with open(f"debug_case_{url.split('/')[-2]}.html", "w", encoding="utf-8") as f:
#         f.write(str(soup))
#     logger.info(f"Saved case HTML to debug_case_{url.split('/')[-2]}.html")

#     try:
#         # Extract title
#         h2_tags = soup.find_all("h2")
#         title = "Unknown Title"
#         if len(h2_tags) >= 2:
#             title = h2_tags[1].text.strip()  # Second h2 tag should be the case title
#         logger.info(f"Extracted title: {title}")

#         # Extract date from the title (e.g., "on 16 July, 1991")
#         date = "Unknown Date"
#         if title != "Unknown Title":
#             date_match = re.search(r"on (\d{1,2} \w+, \d{4})", title)
#             if date_match:
#                 date = date_match.group(1)
#         logger.info(f"Extracted date: {date}")

#         # Extract content from div.judgment-body
#         content_div = soup.find("div", class_="judgment-body")
#         if content_div:
#             paragraphs = content_div.find_all("p")
#             logger.info(f"Found {len(paragraphs)} paragraphs in content div")
#             content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
#         else:
#             logger.warning("Content div not found, trying alternative selectors")
#             # Fallback: Try div.judgement
#             content_div = soup.find("div", class_="judgement")
#             if content_div:
#                 paragraphs = content_div.find_all("p")
#                 logger.info(f"Found {len(paragraphs)} paragraphs in alternative content div")
#                 content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
#             else:
#                 # Fallback: Extract all paragraphs, but filter out advertisement text
#                 paragraphs = soup.find_all("p")
#                 logger.info(f"Found {len(paragraphs)} paragraphs in entire page")
#                 content_parts = []
#                 for p in paragraphs:
#                     text = p.text.strip()
#                     if text and "Take notes as you read a judgment" not in text and "Premium Member Services" not in text:
#                         content_parts.append(text)
#                 content = " ".join(content_parts)

#         if not content or content == "":
#             logger.warning(f"No content found for case: {url}")
#             return None

#         case_data = {
#             "url": url,
#             "title": title,
#             "date": date,
#             "content": content
#         }
#         return case_data
#     except Exception as e:
#         logger.error(f"Error parsing case {url}: {e}")
#         return None
def scrape_case(url):
    """
    Scrape a single case page from IndianKanoon.
    Args:
        url (str): URL of the case page.
    Returns:
        dict: Scraped case data or None if failed.
    """
    soup = fetch_page(url, use_playwright=True)
    if not soup:
        logger.error(f"Failed to scrape case: {url}")
        return None

    # Save the HTML of the case page for debugging
    # with open(f"debug_case_{url.split('/')[-2]}.html", "w", encoding="utf-8") as f:
    #     f.write(str(soup))
    # logger.info(f"Saved case HTML to debug_case_{url.split('/')[-2]}.html")

    try:
        # Extract title
        h2_tags = soup.find_all("h2")
        title = "Unknown Title"
        if len(h2_tags) >= 2:
            title = h2_tags[1].text.strip()  # Second h2 tag should be the case title
        logger.info(f"Extracted title: {title}")

        # Extract date from the title (e.g., "on 16 July, 1991")
        date = "Unknown Date"
        if title != "Unknown Title":
            date_match = re.search(r"on (\d{1,2} \w+, \d{4})", title)
            if date_match:
                date = date_match.group(1)
        logger.info(f"Extracted date: {date}")

        # Try new selector: div.judgment-text
        content_div = soup.find("div", class_="judgment-text")
        if content_div:
            paragraphs = content_div.find_all("p")
            logger.info(f"Found {len(paragraphs)} paragraphs in content div")
            content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
        else:
            logger.warning("Content div not found, trying alternative selectors")
            # Fallback: Try div.judgment-body
            content_div = soup.find("div", class_="judgment-body")
            if content_div:
                paragraphs = content_div.find_all("p")
                logger.info(f"Found {len(paragraphs)} paragraphs in alternative content div")
                content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
            else:
                # Fallback: Try div.judgement
                content_div = soup.find("div", class_="judgement")
                if content_div:
                    paragraphs = content_div.find_all("p")
                    logger.info(f"Found {len(paragraphs)} paragraphs in alternative content div")
                    content = " ".join([p.text.strip() for p in paragraphs if p.text.strip()])
                else:
                    # Fallback: Extract all paragraphs, filter out ad text
                    paragraphs = soup.find_all("p")
                    logger.info(f"Found {len(paragraphs)} paragraphs in entire page")
                    content_parts = []
                    for p in paragraphs:
                        text = p.text.strip()
                        if text and "Take notes as you read a judgment" not in text and "Premium Member Services" not in text:
                            content_parts.append(text)
                    content = " ".join(content_parts)

        if not content or content == "":
            logger.warning(f"No content found for case: {url}")
            return None

        case_data = {
            "url": url,
            "title": title,
            "date": date,
            "content": content
        }
        return case_data
    
    except Exception as e:
        logger.error(f"Error parsing case {url}: {e}")
        return None

def scrape_cases(search_url, max_cases=10, output_dir="data/raw/indkanoon"):
    """
    Scrape multiple cases from an IndianKanoon search page.
    Args:
        search_url (str): URL of the search page.
        max_cases (int): Maximum number of cases to scrape.
        output_dir (str): Directory to save JSON files.
    Returns:
        list: List of scraped case data dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    case_links = get_case_links(search_url, max_cases)
    if not case_links:
        logger.error(f"No case links found for {search_url}. Check debug_search.html for HTML structure.")
        return []

    cases_data = []
    for idx, url in enumerate(case_links):
        logger.info(f"Scraping case: {url}")
        case_data = scrape_case(url)
        if case_data:
            json_filename = f"case_{idx}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved case data to {json_path}")
            cases_data.append(case_data)
        else:
            logger.warning(f"Skipping case {url} due to scraping failure")

    return cases_data

if __name__ == "__main__":
    search_url = "https://indiankanoon.org/search/?formInput=ipc%20302"
    cases = scrape_cases(search_url, max_cases=10)
    if cases:
        print("Scrapped")
        # for case in cases:
        #     print(json.dumps(case, indent=4))
    else:
        print("No cases scraped. Check app.log and debug_search.html for details.")