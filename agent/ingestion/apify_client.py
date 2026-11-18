"""Apify integration client for scraping financial social forums and Telegram channels."""
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional
import pandas as pd
from agent.ingestion.base import BaseSocialScraper, SocialPost
from agent.ingestion.mock_data import SyntheticFeedGenerator
from agent.utils.logger import setup_logger

logger = setup_logger("ApifyScraper")


class ApifySocialScraper(BaseSocialScraper):
    """Scrapes community forums and social channels via Apify actors with graceful fallback."""

    def __init__(
        self,
        api_token: Optional[str] = None,
        use_fallback: bool = True,
        actor_id: str = "apify/telegram-scraper"
    ):
        self.api_token = api_token or os.getenv("APIFY_TOKEN", "")
        self.use_fallback = use_fallback
        self.actor_id = actor_id
        self._client = None

        if self.api_token and not self.api_token.startswith("apify_api_your_token"):
            try:
                from apify_client import ApifyClient
                self._client = ApifyClient(self.api_token)
                logger.info(f"Initialized ApifyClient with actor: {self.actor_id}")
            except Exception as e:
                logger.warning(f"Could not initialize ApifyClient: {e}. Falling back to fallback generator.")
        else:
            logger.info("No valid APIFY_TOKEN provided; pipeline running in high-fidelity simulation mode.")

    def fetch_posts(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        cashtags: List[str],
        days_back: int = 10
    ) -> List[SocialPost]:
        """Fetch and normalize social media posts for an asset."""
        posts: List[SocialPost] = []

        if self._client:
            try:
                logger.info(f"Triggering Apify actor '{self.actor_id}' for {ticker} ({asset_name})...")
                # Construct search query from keywords and cashtags
                search_query = " OR ".join([f'"{kw}"' for kw in keywords[:2]] + cashtags[:2])
                
                run_input = {
                    "search": search_query,
                    "maxItems": 50,
                    "dateFrom": (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                }
                
                run = self._client.actor(self.actor_id).call(run_input=run_input, timeout_secs=60)
                dataset_items = self._client.dataset(run["defaultDatasetId"]).list_items().items
                
                for item in dataset_items:
                    # Generic normalization across different community scrapers
                    raw_text = item.get("text") or item.get("message") or item.get("content") or item.get("title", "")
                    if not raw_text or len(raw_text.strip()) < 5:
                        continue
                        
                    raw_time = item.get("date") or item.get("timestamp") or item.get("created_at")
                    try:
                        ts = pd.to_datetime(raw_time, utc=True).to_pydatetime() if raw_time else datetime.now(timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)

                    post_id = str(item.get("id") or item.get("postId") or hash(raw_text))
                    platform = item.get("platform") or self._infer_platform(self.actor_id)

                    posts.append(
                        SocialPost(
                            post_id=post_id,
                            ticker=ticker,
                            platform=platform,
                            timestamp=ts,
                            text=str(raw_text),
                            author=item.get("author") or item.get("username", "retail_trader"),
                            author_followers=int(item.get("followers", item.get("authorFollowers", 0)) or 0),
                            likes=int(item.get("likes", item.get("upvotes", 0)) or 0),
                            reposts=int(item.get("reposts", item.get("shares", 0)) or 0),
                            views=int(item.get("views", 0) or 0),
                            url=item.get("url"),
                            raw_metadata=item
                        )
                    )
                logger.info(f"Successfully scraped {len(posts)} posts from Apify for {ticker}.")
            except Exception as e:
                logger.warning(f"Apify actor execution failed for {ticker}: {e}. Engaging graceful fallback.")

        # Fallback to high-fidelity feed if empty or token missing
        if not posts and self.use_fallback:
            exchange = "krx" if ticker.endswith(".KS") else "hkex"
            posts = SyntheticFeedGenerator.generate_social_posts(
                ticker=ticker,
                asset_name=asset_name,
                days_back=days_back,
                exchange=exchange
            )

        return posts

    def _infer_platform(self, actor_id: str) -> str:
        """Infer platform name from actor ID string."""
        actor_lower = actor_id.lower()
        if "telegram" in actor_lower:
            return "telegram"
        if "lihkg" in actor_lower:
            return "lihkg"
        if "futu" in actor_lower or "moomoo" in actor_lower:
            return "futu"
        if "aastocks" in actor_lower or "etnet" in actor_lower:
            return "aastocks"
        if "naver" in actor_lower:
            return "naver_finance"
        if "daum" in actor_lower:
            return "daum_finance"
        if "blind" in actor_lower:
            return "blind"
        return "community_forum"

