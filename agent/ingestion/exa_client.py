"""Exa.com integration client for financial news retrieval and sentiment ingestion."""
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import pandas as pd
from agent.ingestion.base import BaseNewsClient, NewsArticle
from agent.ingestion.mock_data import SyntheticFeedGenerator
from agent.utils.logger import setup_logger

logger = setup_logger("ExaNewsClient")


class ExaNewsClient(BaseNewsClient):
    """Retrieves authoritative news articles via Exa API with keyword and date filtering."""

    def __init__(self, api_key: Optional[str] = None, use_fallback: bool = True):
        self.api_key = api_key or os.getenv("EXA_API_KEY", "")
        self.use_fallback = use_fallback
        self._client = None

        if self.api_key and not self.api_key.startswith("exa_api_key_your"):
            try:
                from exa_py import Exa
                self._client = Exa(api_key=self.api_key)
                logger.info("Initialized Exa news search client.")
            except Exception as e:
                logger.warning(f"Could not initialize Exa client: {e}. Fallback enabled.")
        else:
            logger.info("No EXA_API_KEY configured; utilizing high-fidelity news simulation mode.")

    def fetch_news(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        days_back: int = 10,
        max_results: int = 20
    ) -> List[NewsArticle]:
        """Fetch recent and relevant news articles for an asset."""
        articles: List[NewsArticle] = []

        if self._client:
            try:
                start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
                query = f"{asset_name} ({ticker}) stock earnings revenue business news"
                logger.info(f"Querying Exa.com for '{query}' since {start_date}...")

                if hasattr(self._client, "search"):
                    results = self._client.search(
                        query=query,
                        type="neural",
                        use_autoprompt=True,
                        num_results=max_results,
                        start_published_date=start_date,
                        highlights=True
                    )
                else:
                    results = self._client.search_and_contents(
                        query=query,
                        type="neural",
                        use_autoprompt=True,
                        num_results=max_results,
                        start_published_date=start_date,
                        highlights=True
                    )

                for item in getattr(results, "results", []):
                    pub_date_raw = getattr(item, "published_date", None)
                    try:
                        ts = pd.to_datetime(pub_date_raw, utc=True).to_pydatetime() if pub_date_raw else datetime.now(timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)

                    snippet = ""
                    highlights = getattr(item, "highlights", [])
                    if highlights:
                        snippet = " ".join(highlights[:2])
                    elif hasattr(item, "text") and item.text:
                        snippet = item.text[:300]

                    title = getattr(item, "title", f"{asset_name} Market Update") or f"{asset_name} Market Update"
                    url = getattr(item, "url", "https://exa.ai")
                    art_id = getattr(item, "id", str(hash(url)))
                    score = getattr(item, "score", 0.95) or 0.95

                    # Infer publisher from domain
                    publisher = "Financial News"
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        if domain:
                            publisher = domain.replace("www.", "").split(".")[0].capitalize()
                    except Exception:
                        pass

                    articles.append(
                        NewsArticle(
                            article_id=art_id,
                            ticker=ticker,
                            source_engine="exa",
                            timestamp=ts,
                            title=title,
                            snippet=snippet,
                            url=url,
                            publisher=publisher,
                            relevance_score=float(score)
                        )
                    )
                logger.info(f"Exa retrieved {len(articles)} articles for {ticker}.")
            except Exception as e:
                logger.warning(f"Exa search failed for {ticker}: {e}. Engaging fallback.")

        # Fallback to high-fidelity feed if empty or token missing
        if not articles and self.use_fallback:
            exchange = "krx" if ticker.endswith(".KS") else "hkex"
            articles = SyntheticFeedGenerator.generate_news_articles(
                ticker=ticker,
                asset_name=asset_name,
                days_back=days_back,
                exchange=exchange
            )

        return articles
