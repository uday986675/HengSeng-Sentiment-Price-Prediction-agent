"""Base data classes and abstract interfaces for sentiment and price ingestion."""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SocialPost(BaseModel):
    """Normalized social media post representation."""
    post_id: str
    ticker: str
    platform: str # e.g. "telegram", "lihkg", "futu", "naver_finance", "daum", "blind"
    timestamp: datetime # UTC-aligned
    text: str
    author: Optional[str] = "anonymous"
    author_followers: int = 0
    likes: int = 0
    reposts: int = 0
    views: int = 0
    url: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def total_engagement(self) -> int:
        """Composite engagement metric."""
        return int(self.likes * 2 + self.reposts * 3 + max(0, self.views // 10))


class NewsArticle(BaseModel):
    """Normalized news item representation from Exa / Tavily."""
    article_id: str
    ticker: str
    source_engine: str # "exa" or "tavily"
    timestamp: datetime # UTC-aligned
    title: str
    snippet: str
    url: str
    publisher: Optional[str] = "Financial News"
    relevance_score: float = 1.0


class PriceBar(BaseModel):
    """Standardized daily price bar aligned to UTC."""
    ticker: str
    timestamp: datetime # UTC-aligned midnight date
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float


class BaseSocialScraper(ABC):
    """Abstract interface for social media scrapers."""

    @abstractmethod
    def fetch_posts(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        cashtags: List[str],
        days_back: int = 10
    ) -> List[SocialPost]:
        """Fetch and normalize social media posts for an asset."""
        pass


class BaseNewsClient(ABC):
    """Abstract interface for news retrieval engines."""

    @abstractmethod
    def fetch_news(
        self,
        ticker: str,
        asset_name: str,
        keywords: List[str],
        days_back: int = 10,
        max_results: int = 20
    ) -> List[NewsArticle]:
        """Fetch recent and relevant news articles for an asset."""
        pass


class BasePriceClient(ABC):
    """Abstract interface for market price ingestion."""

    @abstractmethod
    def fetch_prices(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceBar]:
        """Fetch daily OHLCV bars aligned to UTC."""
        pass

