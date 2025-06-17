# # src/scraper/utils.py
# import requests
# from bs4 import BeautifulSoup
# import time
# import os
# from playwright.sync_api import sync_playwright
# from src.utils.logger import logger

# def fetch_page(url, retries=3, delay=2, use_playwright=False):
#     """
#     Fetch a webpage with retries and delay to avoid rate limits.
#     Args:
#         url (str): URL to fetch.
#         retries (int): Number of retry attempts.
#         delay (int): Seconds to wait between retries.
#         use_playwright (bool): Use Playwright for dynamic content.
#     Returns:
#         BeautifulSoup object or None if failed.
#     """
#     if not use_playwright:
#         headers = {
#             "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
#         }
#         for attempt in range(retries):
#             try:
#                 response = requests.get(url, headers=headers, timeout=10)
#                 response.raise_for_status()
#                 logger.info(f"Fetched URL: {url}")
#                 with open("debug_search.html", "w", encoding="utf-8") as f:
#                     f.write(response.text)
#                 logger.info(f"Saved HTML to debug_search.html")
#                 time.sleep(delay)
#                 return BeautifulSoup(response.text, "html.parser")
#             except requests.RequestException as e:
#                 logger.error(f"Error fetching {url}: {e}")
#                 if attempt < retries - 1:
#                     logger.info(f"Retrying {url} after {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     logger.error(f"Failed to fetch {url} after {retries} attempts")
#                     return None

#     # Use Playwright for dynamic content
#     logger.info(f"Using Playwright to fetch {url}")
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
#         page.goto(url)
#         try:
#             # Wait for any <a> tag with href containing "/doc/"
#             page.wait_for_selector("a[href*='/doc/']", timeout=15000)  # Wait up to 15 seconds
#             logger.info("Found case links in the page")
#         except Exception as e:
#             logger.error(f"Timeout waiting for case links: {e}")
#         html = page.content()
#         browser.close()
#         with open("debug_search.html", "w", encoding="utf-8") as f:
#             f.write(html)
#         logger.info(f"Saved HTML to debug_search.html")
#         time.sleep(delay)
#         return BeautifulSoup(html, "html.parser")

# def get_case_links(search_url, max_cases=10):
#     """
#     Extract case URLs from an IndianKanoon search page.
#     Args:
#         search_url (str): URL of the search page.
#         max_cases (int): Maximum number of case links to extract.
#     Returns:
#         list: List of case URLs.
#     """
#     soup = fetch_page(search_url, use_playwright=True)
#     if not soup:
#         return []

#     case_links = []
#     try:
#         links = soup.find_all("a", href=True)
#         for link in links:
#             href = link.get("href")
#             if href and "/doc/" in href and href.startswith("/doc/"):
#                 full_url = f"https://indiankanoon.org{href}"
#                 case_links.append(full_url)
#                 if len(case_links) >= max_cases:
#                     break
#         # Remove duplicates
#         case_links = list(dict.fromkeys(case_links))[:max_cases]
#         logger.info(f"Extracted {len(case_links)} case links from {search_url}")
#         return case_links
#     except Exception as e:
#         logger.error(f"Error extracting case links from {search_url}: {e}")
#         return []


import requests
from bs4 import BeautifulSoup
import time
import os
import re
from playwright.sync_api import sync_playwright
from src.utils.logger import logger

def fetch_page(url, retries=3, delay=2, use_playwright=False):
    """
    Fetch a webpage with retries and delay to avoid rate limits.
    Args:
        url (str): URL to fetch.
        retries (int): Number of retry attempts.
        delay (int): Seconds to wait between retries.
        use_playwright (bool): Use Playwright for dynamic content.
    Returns:
        BeautifulSoup object or None if failed.
    """
    if not use_playwright:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
        }
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                logger.info(f"Fetched URL: {url}")
                with open("debug_search.html", "w", encoding="utf-8") as f:
                    f.write(response.text)
                logger.info(f"Saved HTML to debug_search.html")
                time.sleep(delay)
                return BeautifulSoup(response.text, "html.parser")
            except requests.RequestException as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying {url} after {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    # Use Playwright for dynamic content
    logger.info(f"Using Playwright to fetch {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        try:
            page.wait_for_timeout(20000)  # Wait 20 seconds for initial load
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # Scroll to trigger dynamic content
            page.wait_for_timeout(5000)  # Wait after scrolling
            page.wait_for_selector("a[href*='/doc/']", timeout=15000)  # Wait up to 15 seconds
            logger.info("Found case links in the page")
        except Exception as e:
            logger.error(f"Timeout waiting for case links: {e}")
        html = page.content()
        browser.close()
        with open("debug_search.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Saved HTML to debug_search.html")
        time.sleep(delay)
        return BeautifulSoup(html, "html.parser")

def get_case_links(search_url, max_cases=10):
    """
    Extract case URLs from an IndianKanoon search page.
    Args:
        search_url (str): URL of the search page.
        max_cases (int): Maximum number of case links to extract.
    Returns:
        list: List of case URLs.
    """
    soup = fetch_page(search_url, use_playwright=True)
    if not soup:
        return []

    case_links = []
    try:
        results = soup.find_all("div", class_="result")
        logger.info(f"Found {len(results)} result entries")
        for idx, result in enumerate(results):
            link = result.find("a", href=True)
            if link:
                href = link.get("href")
                title = link.text.strip()
                logger.info(f"Processing result {idx + 1}: Title={title}, Href={href}")
                # Skip statutes (e.g., entries with "Section" or "Act" in the title)
                if "Section" in title or "Act" in title:
                    logger.info(f"Skipping statute link: {title} ({href})")
                    continue
                # Handle both /doc/ and /docfragment/ patterns
                doc_match = re.match(r"/docfragment/(\d+)/", href) or re.match(r"/doc/(\d+)/", href)
                if doc_match:
                    doc_id = doc_match.group(1)
                    full_url = f"https://indiankanoon.org/doc/{doc_id}/"
                    case_links.append(full_url)
                    logger.info(f"Added case link: {full_url}")
                else:
                    logger.warning(f"Link does not match case pattern: {href}")
                if len(case_links) >= max_cases:
                    break
        # Remove duplicates
        case_links = list(dict.fromkeys(case_links))[:max_cases]
        logger.info(f"Extracted {len(case_links)} case links from {search_url}")
        return case_links
    except Exception as e:
        logger.error(f"Error extracting case links from {search_url}: {e}")
        return []