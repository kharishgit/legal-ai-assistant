import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import requests
from src.utils.logger import logger

def fetch_news_articles(api_key, query="Indian Penal Code 302", max_articles=10, output_dir="data/raw/newsapi"):
    """
    Fetch news articles using NewsAPI.
    Args:
        api_key (str): NewsAPI key.
        query (str): Search query.
        max_articles (int): Maximum number of articles to fetch.
        output_dir (str): Directory to save JSON files.
    Returns:
        list: List of article data dictionaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    articles_data = []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": api_key,
        "pageSize": max_articles,
        # Removed domains to broaden the search
    }
    
    try:
        logger.info(f"Fetching articles for query: {query}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
            
        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} articles from NewsAPI")
        
        for idx, article in enumerate(articles[:max_articles]):
            article_data = {
                "url": article.get("url", ""),
                "title": article.get("title", "Unknown Title"),
                "date": article.get("publishedAt", "Unknown Date"),
                "content": article.get("content", "No content available") or "No content available"
            }
            
            # Save to JSON
            json_filename = f"article_{idx}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved article data to {json_path}")
            articles_data.append(article_data)
            
        return articles_data
        
    except requests.RequestException as e:
        logger.error(f"Error fetching articles from NewsAPI: {e}")
        return []

if __name__ == "__main__":
    # Replace with your NewsAPI key
    API_KEY = "e422b2d0fd87478c9dec46ccd1405797"
    articles = fetch_news_articles(API_KEY, query="Indian Penal Code 302", max_articles=10)
    for article in articles:
        print(json.dumps(article, indent=4))