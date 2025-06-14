import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from src.utils.logger import logger

def fetch_page_with_playwright(url, wait_selector=None, timeout=60000, retries=2):
    """
    Fetch a webpage using Playwright to handle dynamic content with retries.
    Args:
        url (str): URL to fetch.
        wait_selector (str): CSS selector to wait for (optional).
        timeout (int): Timeout in milliseconds.
        retries (int): Number of retry attempts.
    Returns:
        BeautifulSoup object or None if failed.
    """
    for attempt in range(retries + 1):
        logger.info(f"Attempt {attempt + 1}: Using Playwright to fetch {url}")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    viewport={"width": 1280, "height": 720},
                    accept_downloads=True
                )
                page = context.new_page()
                page.set_extra_http_headers({
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
                })
                response = page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                logger.info(f"Navigated to URL: {page.url}")

                page.wait_for_timeout(20000)  # Wait 20 seconds for JavaScript to render
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # Scroll to trigger dynamic content
                page.wait_for_timeout(5000)  # Wait after scrolling

                if wait_selector:
                    try:
                        elements = page.query_selector_all(wait_selector)
                        if elements:
                            logger.info(f"Found {len(elements)} elements matching selector {wait_selector}")
                            for i, element in enumerate(elements[:5]):
                                href = element.get_attribute("href") if "href" in wait_selector else None
                                logger.info(f"Element {i+1}: href={href}")
                        else:
                            logger.warning(f"No elements found for selector {wait_selector}")
                        page.wait_for_selector(wait_selector, timeout=timeout, state="attached")
                        logger.info(f"Selector {wait_selector} is present in DOM")
                    except Exception as e:
                        logger.warning(f"Selector {wait_selector} not found, proceeding with page content: {e}")

                html = page.content()
                with open("debug_news.html", "w", encoding="utf-8") as f:
                    f.write(html)
                logger.info(f"Saved HTML to debug_news.html")
                return BeautifulSoup(html, "html.parser")
        except Exception as e:
            logger.error(f"Error fetching {url} on attempt {attempt + 1}: {e}")
            if attempt < retries:
                logger.info(f"Retrying {url} after a delay...")
                if page:
                    page.wait_for_timeout(2000)
            else:
                logger.error(f"Failed to fetch {url} after {retries + 1} attempts")
                return None

def get_article_links(search_url, max_articles=10):
    """
    Extract article URLs from a news search page (Legally India).
    Args:
        search_url (str): URL of the search page.
        max_articles (int): Maximum number of article links to extract.
    Returns:
        list: List of article URLs.
    """
    # Use a general selector to ensure the page loads
    soup = fetch_page_with_playwright(search_url, wait_selector="body")
    if not soup:
        return []

    article_links = []
    try:
        # Adjust selector based on Legally India's structure (to be confirmed with debug HTML)
        # Assuming articles are in <h2> or <h3> tags with links
        headers = soup.find_all(["h2", "h3"])
        for header in headers:
            link = header.find("a", href=True)
            if link:
                href = link.get("href")
                if href and href.startswith("https://www.legallyindia.com/"):
                    article_links.append(href)
                    if len(article_links) >= max_articles:
                        break
        article_links = list(dict.fromkeys(article_links))[:max_articles]
        logger.info(f"Extracted {len(article_links)} article links from {search_url}")
        return article_links
    except Exception as e:
        logger.error(f"Error extracting article links from {search_url}: {e}")
        return []

def scrape_article(url):
    """
    Scrape a single news article page (Legally India).
    Args:
        url (str): URL of the article page.
    Returns:
        dict: Scraped article data or None if failed.
    """
    soup = fetch_page_with_playwright(url, wait_selector="body")
    if not soup:
        logger.error(f"Failed to scrape article: {url}")
        return None

    try:
        # Adjust selectors based on Legally India's structure (to be confirmed with debug HTML)
        title_tag = soup.find("h1")
        title = title_tag.text.strip() if title_tag else "Unknown Title"

        date_tag = soup.find("time") or soup.find("span", class_="date")
        date = date_tag.text.strip() if date_tag else "Unknown Date"

        content_div = soup.find("div", class_="content") or soup.find("article")
        content = " ".join([p.text.strip() for p in content_div.find_all("p")]) if content_div else "No content found"

        if content == "No content found":
            logger.warning(f"No content found for article: {url}")
            return None

        article_data = {
            "url": url,
            "title": title,
            "date": date,
            "content": content
        }
        return article_data
    except Exception as e:
        logger.error(f"Error parsing article {url}: {e}")
        return None

def scrape_news_articles(search_url, max_articles=10, output_dir="data/raw/legallyindia"):
    """
    Scrape multiple articles from a news search page.
    Args:
        search_url (str): URL of the search page.
        max_articles (int): Maximum number of articles to scrape.
        output_dir (str): Directory to save JSON files.
    Returns:
        list: List of scraped article data dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    article_links = get_article_links(search_url, max_articles)
    if not article_links:
        logger.error(f"No article links found for {search_url}")
        return []

    articles_data = []
    for idx, url in enumerate(article_links):
        logger.info(f"Scraping article: {url}")
        article_data = scrape_article(url)
        if article_data:
            json_filename = f"article_{idx}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved article data to {json_path}")
            articles_data.append(article_data)
        else:
            logger.warning(f"Skipping article {url} due to scraping failure")

    return articles_data

# Test the scraper with Legally India
if __name__ == "__main__":
    search_url = "https://www.legallyindia.com/search?q=IPC+302"
    articles = scrape_news_articles(search_url, max_articles=10)
    for article in articles:
        print(json.dumps(article, indent=4))