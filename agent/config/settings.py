"""Configuration schema and loading logic using Pydantic."""
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

class AssetConfig(BaseModel):
    ticker: str
    name: str
    keywords: List[str]
    cashtags: List[str]

class ApifyActorConfig(BaseModel):
    name: str
    actor_id: str
    description: str
    default_run_input: Dict = Field(default_factory=dict)
    enabled: bool = True

class ExaConfig(BaseModel):
    num_results_per_asset: int = 25
    use_autoprompt: bool = True
    days_back: int = 10
    highlights: bool = True

class TavilyConfig(BaseModel):
    max_results: int = 15
    search_depth: str = "advanced"
    include_raw_content: bool = False

class ExchangeConfig(BaseModel):
    exchange_id: Literal["hkex", "krx"]
    exchange_name: str
    currency: str
    timezone: str = "UTC"
    market_open_utc: str = "01:30"
    market_close_utc: str = "08:00"
    assets: List[AssetConfig]
    social_sources: List[ApifyActorConfig]

class ModelConfig(BaseModel):
    lookback_days: int = 90
    forecast_horizon: int = 1
    cv_n_splits: int = 5
    test_size_days: int = 20
    model_type: Literal["ridge", "random_forest", "gradient_boosting", "ensemble"] = "ensemble"
    random_state: int = 42

class PipelineConfig(BaseModel):
    exchange: Literal["hkex", "krx"] = "hkex"
    history_days: int = 30
    sentiment_lookback_days: int = 10
    cache_dir: Path = Path("data/cache")
    processed_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    exchanges: Dict[str, ExchangeConfig]
    exa: ExaConfig = Field(default_factory=ExaConfig)
    tavily: TavilyConfig = Field(default_factory=TavilyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # API credentials read from environment
    apify_token: Optional[str] = Field(default_factory=lambda: os.getenv("APIFY_TOKEN", ""))
    exa_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    tavily_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    use_synthetic_fallback: bool = Field(default_factory=lambda: os.getenv("USE_SYNTHETIC_FALLBACK", "true").lower() == "true")

def get_default_config(exchange: Literal["hkex", "krx"] = "hkex") -> PipelineConfig:
    """Returns default configuration with complete assets, Apify actors, and parameters."""
    hkex_assets = [
        AssetConfig(
            ticker="0700.HK",
            name="Tencent Holdings",
            keywords=["Tencent", "腾讯", "騰訊", "WeChat", "0700"],
            cashtags=["#0700", "$0700", "#Tencent"]
        ),
        AssetConfig(
            ticker="9988.HK",
            name="Alibaba Group",
            keywords=["Alibaba", "阿里巴巴", "阿里", "Taobao", "9988"],
            cashtags=["#9988", "$9988", "#BABA"]
        ),
        AssetConfig(
            ticker="0005.HK",
            name="HSBC Holdings",
            keywords=["HSBC", "汇丰", "滙豐", "0005"],
            cashtags=["#0005", "$0005", "#HSBC"]
        ),
        AssetConfig(
            ticker="3690.HK",
            name="Meituan",
            keywords=["Meituan", "美团", "美團", "KeeTa", "3690"],
            cashtags=["#3690", "$3690", "#Meituan"]
        ),
        AssetConfig(
            ticker="9618.HK",
            name="JD.com",
            keywords=["JD.com", "京东", "京東", "9618"],
            cashtags=["#9618", "$9618", "#JD"]
        )
    ]
    
    krx_assets = [
        AssetConfig(
            ticker="005930.KS",
            name="Samsung Electronics",
            keywords=["Samsung Electronics", "삼성전자", "삼전", "005930"],
            cashtags=["#005930", "$005930", "#Samsung"]
        ),
        AssetConfig(
            ticker="000660.KS",
            name="SK Hynix",
            keywords=["SK Hynix", "SK하이닉스", "하이닉스", "000660"],
            cashtags=["#000660", "$000660", "#Hynix"]
        ),
        AssetConfig(
            ticker="035420.KS",
            name="NAVER Corporation",
            keywords=["NAVER", "네이버", "035420"],
            cashtags=["#035420", "$035420", "#NAVER"]
        ),
        AssetConfig(
            ticker="035720.KS",
            name="Kakao",
            keywords=["Kakao", "카카오", "035720"],
            cashtags=["#035720", "$035720", "#Kakao"]
        ),
        AssetConfig(
            ticker="005380.KS",
            name="Hyundai Motor",
            keywords=["Hyundai Motor", "현대차", "현대자동차", "005380"],
            cashtags=["#005380", "$005380", "#Hyundai"]
        )
    ]

    hkex_social_sources = [
        ApifyActorConfig(
            name="telegram_hk_markets",
            actor_id="apify/telegram-scraper",
            description="Hong Kong retail trading communities & financial telegram channels",
            default_run_input={"maxMessages": 100, "onlyNewMessages": False}
        ),
        ApifyActorConfig(
            name="lihkg_finance_forum",
            actor_id="actor-registry/lihkg-crawler",
            description="LIHKG Hong Kong stock & financial debate threads",
            default_run_input={"category": "finance", "maxPosts": 60}
        ),
        ApifyActorConfig(
            name="futu_moomoo_community",
            actor_id="apify/web-scraper",
            description="Futu / Moomoo retail investor forum & quote comments scraper",
            default_run_input={"pageFunction": "futu_scraper_v1", "maxItems": 50}
        ),
        ApifyActorConfig(
            name="aastocks_etnet_comments",
            actor_id="apify/cheerio-scraper",
            description="AAStocks and ETNet market commentary & retail quotes scraper",
            default_run_input={"maxRequestsPerCrawl": 50}
        )
    ]
    
    krx_social_sources = [
        ApifyActorConfig(
            name="naver_finance_board",
            actor_id="apify/web-scraper",
            description="Naver Finance Stock Discussion Boards (종목토론실)",
            default_run_input={"target": "finance.naver.com/item/board.naver", "maxPosts": 60}
        ),
        ApifyActorConfig(
            name="daum_finance_forum",
            actor_id="apify/cheerio-scraper",
            description="Daum Finance community stock forums",
            default_run_input={"target": "finance.daum.net", "maxPosts": 50}
        ),
        ApifyActorConfig(
            name="kakaotalk_telegram_kr",
            actor_id="apify/telegram-scraper",
            description="South Korea equity trading open groups & telegram alert rooms",
            default_run_input={"maxMessages": 100}
        ),
        ApifyActorConfig(
            name="blind_kr_insider",
            actor_id="apify/puppeteer-scraper",
            description="Blind verified employee sentiment & chaebol enterprise discussions",
            default_run_input={"maxItems": 40}
        )
    ]

    exchanges = {
        "hkex": ExchangeConfig(
            exchange_id="hkex",
            exchange_name="Hong Kong Stock Exchange",
            currency="HKD",
            timezone="UTC",
            assets=hkex_assets,
            social_sources=hkex_social_sources
        ),
        "krx": ExchangeConfig(
            exchange_id="krx",
            exchange_name="Korea Exchange",
            currency="KRW",
            timezone="UTC",
            assets=krx_assets,
            social_sources=krx_social_sources
        )
    }

    return PipelineConfig(
        exchange=exchange,
        exchanges=exchanges
    )
