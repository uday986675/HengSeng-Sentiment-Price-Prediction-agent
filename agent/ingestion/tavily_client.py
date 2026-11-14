"""Tavily search client serving as an alternate ingestion backend behind the standard interfaces."""
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import requests
import pandas as pd
from agent.ingestion.base import BaseNewsClient, BaseSocialScraper, NewsArticle, SocialPost
from agent.ingestion.mock_data import SyntheticFeedGenerator
from agent.utils.logger import setup_logger

logger = setup_logger("TavilyClient")


class TavilyIngestionClient(BaseNewsClient, BaseSocialScraper):
    """Alternate ingestion client utilizing Tavily Search API for both news and social discussions."""

    def __init__(self, api_key: Optional[str] = None, use_fallback: bool = True):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.use_fallback = use_fallback
        self.endpoint = "https://api.tavily.com/search"

    def fetch_news(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        days_back: int = 10,
        max_results: int = 20
    ) -> List[NewsArticle]:
        """Fetch news articles using Tavily Search API."""
        articles: List[NewsArticle] = []

        if self.api_key and not self.api_key.startswith("tvly-your_key"):
            try:
                payload = {
                    "api_key": self.api_key,
                    "query": f"{asset_name} {ticker} stock market financial news",
                    "search_depth": "advanced",
                    "include_answer": False,
                    "max_results": max_results,
                    "topic": "news",
                    "days": days_back
                }
                logger.info(f"Querying Tavily News for {ticker} ({asset_name})...")
                resp = requests.post(self.endpoint, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("results", []):
                        published_date = item.get("published_date")
                        try:
                            ts = pd.to_datetime(published_date, utc=True).to_pydatetime() if published_date else datetime.now(timezone.utc)
                        except Exception:
                            ts = datetime.now(timezone.utc)

                        url = item.get("url", "")
                        art_id = str(hash(url))
                        title = item.get("title", f"{asset_name} News")
                        content = item.get("content", "")
                        score = float(item.get("score", 0.90))

                        articles.append(
                            NewsArticle(
                                article_id=art_id,
                                ticker=ticker,
                                source_engine="tavily",
                                timestamp=ts,
                                title=title,
                                snippet=content[:300],
                                url=url,
                                publisher="Tavily News Feed",
                                relevance_score=score
                            )
                        )
                    logger.info(f"Tavily returned {len(articles)} articles for {ticker}.")
            except Exception as e:
                logger.warning(f"Tavily news search failed for {ticker}: {e}.")

        if not articles and self.use_fallback:
            exchange = "krx" if ticker.endswith(".KS") else "hkex"
            articles = SyntheticFeedGenerator.generate_news_articles(
                ticker=ticker,
                asset_name=asset_name,
                days_back=days_back,
                exchange=exchange
            )

        return articles

    def fetch_posts(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        cashtags: List[str],
        days_back: int = 10
    ) -> List[SocialPost]:
        """Fetch forum discussions via Tavily as fallback social scraper."""
        posts: List[SocialPost] = []

        if self.api_key and not self.api_key.startswith("tvly-your_key"):
            try:
                forum_domains = ["lihkg.com", "t.me", "moomoo.com", "futu.com", "naver.com", "daum.net", "teamblind.com"]
                query = f"{asset_name} OR {' OR '.join(cashtags[:2])} forum discussion trader"
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_domains": forum_domains,
                    "max_results": 20,
                    "days": days_back
                }
                logger.info(f"Querying Tavily for social forum discussions on {ticker}...")
                resp = requests.post(self.endpoint, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("results", []):
                        content = item.get("content", "")
                        if not content:
                            continue
                        url = item.get("url", "")
                        ts = datetime.now(timezone.utc)
                        post_id = str(hash(url + content[:20]))

                        posts.append(
                            SocialPost(
                                post_id=post_id,
                                ticker=ticker,
                                platform="tavily_web_forum",
                                timestamp=ts,
                                text=content,
                                author="forum_user",
                                author_followers=500,
                                likes=10,
                                reposts=2,
                                views=150,
                                url=url,
                                raw_metadata={"source": "tavily"}
                            )
                        )
            except Exception as e:
                logger.warning(f"Tavily social discussion search failed for {ticker}: {e}.")

        if not posts and self.use_fallback:
            exchange = "krx" if ticker.endswith(".KS") else "hkex"
            posts = SyntheticFeedGenerator.generate_social_posts(
                ticker=ticker,
                asset_name=asset_name,
                days_back=days_back,
                exchange=exchange
            )

        return posts
