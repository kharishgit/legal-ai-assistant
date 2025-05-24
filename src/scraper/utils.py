# src/scraper/utils.py
import requests
from bs4 import BeautifulSoup
import time
from src.utils.logger import logger

def fetch_page(url, retries=3, delay=2):
    """
    Fetch a webpage with retries and delay to avoid rate limits.
    Args:
        url (str): URL to fetch.
        retries (int): Number of retry attempts.
        delay (int): Seconds to wait between retries.
    Returns:
        BeautifulSoup object or None if failed.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise error for bad status codes
            logger.info(f"Fetched URL: {url}")
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying {url} after {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to fetch {url} after {retries} attempts")
                return None