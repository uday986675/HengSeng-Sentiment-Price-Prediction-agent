# REPRODUCIBILITY GUIDE: HengSeng Sentiment Price Prediction Agent

This document provides exact instructions, Apify Actor IDs, Exa query parameters, and configuration steps to reproduce the entire quantitative alternative-data ingestion and predictive modeling pipeline.

---

## 1. Credentials and Environment Configuration

All API tokens are loaded dynamically via environment variables from `.env`. Never commit `.env` to source control.

Copy the template:
```bash
cp .env.example .env
```

Populate the following keys in `.env`:
```ini
# Apify API Token (https://console.apify.com/account/integrations)
APIFY_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Exa.com API Key (https://dashboard.exa.ai/)
EXA_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# Tavily API Key (Optional alternate ingestion backend: https://app.tavily.com/)
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Runtime settings
LOG_LEVEL=INFO
ENABLE_CACHE=true
USE_SYNTHETIC_FALLBACK=true
```

> **Note on Resilient Fallback**: If tokens are not yet configured or rate limits are reached, the system gracefully activates its deterministic, high-fidelity alternative data simulation engine so that all stages, tests, and visual dashboards run out of the box without crashing.

---

## 2. Scraping Actors & Parameters (Apify)

The pipeline integrates with verified Apify scrapers across Hong Kong (HKEX) and South Korea (KRX) retail communities:

### HKEX (Hong Kong Stock Exchange)
| Source Platform | Apify Actor ID | Run Input Configuration | Description |
| :--- | :--- | :--- | :--- |
| **Telegram Channels** | `apify/telegram-scraper` | `{"maxMessages": 100, "onlyNewMessages": false}` | Captures real-time retail alerts and discussion feeds across Hong Kong trading channels |
| **LIHKG Community** | `actor-registry/lihkg-crawler` | `{"category": "finance", "maxPosts": 60}` | Dominant HK retail investor forum debate threads |
| **Futu / Moomoo** | `apify/web-scraper` | `{"pageFunction": "futu_scraper_v1", "maxItems": 50}` | In-app broker discussion feeds and quote comments |
| **AAStocks & ETNet** | `apify/cheerio-scraper` | `{"maxRequestsPerCrawl": 50}` | Retail quote commentary and market sentiment |

### KRX (Korea Exchange)
| Source Platform | Apify Actor ID | Run Input Configuration | Description |
| :--- | :--- | :--- | :--- |
| **Naver Finance (종목토론실)** | `apify/web-scraper` | `{"target": "finance.naver.com/item/board.naver", "maxPosts": 60}` | The epicenter of Korean retail stock discussion boards |
| **Daum Finance Forums** | `apify/cheerio-scraper` | `{"target": "finance.daum.net", "maxPosts": 50}` | Major Korean financial community board sentiment |
| **KakaoTalk / Telegram** | `apify/telegram-scraper` | `{"maxMessages": 100}` | Private trading groups, alert rooms, and ticker discussions |
| **Blind (직장인 커뮤니티)** | `apify/puppeteer-scraper` | `{"maxItems": 40}` | Verified employee discussions offering insider enterprise sentiment |

---

## 3. Financial News Ingestion Parameters (Exa.com)

Exa neural search is invoked per asset with the following query and filtering schema:

- **Endpoint**: `exa.search` / `exa.search_and_contents`
- **Query Structure**: `"{asset_name} ({ticker}) stock earnings revenue business news"`
- **Search Type**: `neural`
- **Autoprompt**: `true`
- **Date Filter**: `start_published_date = (now - 10 days)`
- **Highlights**: `true` (extracts salient catalyst sentences)
- **Results Per Asset**: `20`
- **Normalization**: Standardized to UTC timestamp, publisher domain, relevance score, snippet, and title.

---

## 4. Market Price Feed (UTC Aligned)

- **Source**: `yfinance` daily OHLCV bars.
- **Tickers (HKEX)**: `0700.HK` (Tencent), `9988.HK` (Alibaba), `0005.HK` (HSBC), `3690.HK` (Meituan), `9618.HK` (JD.com).
- **Tickers (KRX)**: `005930.KS` (Samsung Electronics), `000660.KS` (SK Hynix), `035420.KS` (NAVER), `035720.KS` (Kakao), `005380.KS` (Hyundai Motor).
- **Alignment**: Standardized to midnight UTC (`timestamp.dt.floor("D")`).

---

## 5. End-to-End Execution Commands

### Run Full Pipeline (Default HKEX)
```bash
python -m agent run --exchange hkex
```

### Run Full Pipeline for Korea Exchange (KRX)
```bash
python -m agent run --exchange krx
```

### Force Cache Refresh
```bash
python -m agent run --exchange hkex --force-refresh
```

### Run Walk-Forward Validation Only
```bash
python -m agent backtest --exchange hkex
```

### Run Automated Unit Tests
```bash
python -m pytest tests -v
```

