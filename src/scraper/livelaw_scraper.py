import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))




from playwright.sync_api import sync_playwright
import logging
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def scrape_livelaw() -> List[Dict[str, str]]:
    """Scrape articles from LiveLaw news updates page."""
    articles = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("https://www.livelaw.in/news-updates/", timeout=60000)
            page.wait_for_selector("div.news-item", timeout=10000)  # Adjust selector as needed
            article_elements = page.query_selector_all("div.news-item")
            
            for article in article_elements:
                title = article.query_selector("h2") or article.query_selector("h3")
                content = article.query_selector("p")
                articles.append({
                    "title": title.text_content().strip() if title else "No title",
                    "content": content.text_content().strip() if content else "No content"
                })
            
            logger.info(f"Scraped {len(articles)} articles from LiveLaw")
        except Exception as e:
            logger.error(f"Failed to scrape LiveLaw: {e}")
        finally:
            browser.close()
    
    return articles

if __name__ == "__main__":
    articles = scrape_livelaw()
    for article in articles:
        print(f"Title: {article['title']}, Content: {article['content']}")