"""High-fidelity synthetic data generator for HKEX & KRX social sentiment and news feeds."""
import hashlib
from datetime import datetime, timedelta, timezone
from typing import List
import numpy as np
from agent.ingestion.base import SocialPost, NewsArticle
from agent.utils.logger import setup_logger

logger = setup_logger("SyntheticFeed")


class SyntheticFeedGenerator:
    """Generates realistic alternative social and news sentiment streams."""

    HK_SOCIAL_TEMPLATES = {
        "0700.HK": [
            ("lihkg", "今日騰訊南向資金大買，WeChat視頻號電商GMV翻倍，明天必升破400關口！🚀", 0.85, 340, 52, 4800, 12000),
            ("futu", "0700企穩380重要支撐位，MACD底背馳，外資投行高盛重申買入評級。", 0.70, 180, 24, 2100, 4500),
            ("telegram", "Tencent Cloud AI model announcement scheduled next week. Institutional flows turning net positive.", 0.65, 88, 19, 1300, 980),
            ("aastocks", "大市震盪但騰訊沽空比率降至8%，淡友開始平倉，短線博反彈。", 0.45, 42, 6, 890, 310),
            ("lihkg", "騰訊手遊流水微跌，大戶疑似散貨，追高小心套牢，睇跌至360...", -0.60, 210, 38, 3100, 8500),
            ("futu", "騰訊連續回購10億港元，堅定看好長期股東回報，現價估值極具吸引力。", 0.75, 410, 85, 5900, 16000),
        ],
        "9988.HK": [
            ("futu", "阿里雲分拆進展順利，淘天GMV回暖，看好本季財報超預期！逢低加倉9988！", 0.78, 290, 41, 3800, 8400),
            ("lihkg", "阿里巴巴又被南向資金撈底，電商價格戰趨緩，利潤率企穩。", 0.55, 160, 20, 2400, 5200),
            ("telegram", "BABA buyback pace accelerated this week. Options open interest heavily skewed to $85 calls.", 0.68, 95, 23, 1450, 1100),
            ("aastocks", "阿里雲價格戰可能壓低毛利率，中概板塊情緒依然偏謹慎，留意75元支持。", -0.40, 65, 9, 1100, 420),
            ("lihkg", "9988呢隻死雀幾時先識飛？幾年都係浮浮沉沉，換去美團好過。", -0.50, 190, 32, 2800, 6100),
        ],
        "0005.HK": [
            ("futu", "HSBC宣布啟動30億美元股份回購，息率高達8厘，收息防守首選！", 0.80, 520, 110, 7200, 21000),
            ("lihkg", "匯控業績穩健，淨利息收入雖然見頂，但財富管理費用增長強勁。", 0.45, 140, 18, 2200, 3900),
            ("telegram", "HKMA liquidity remains comfortable. HSBC net interest margin holding up better than consensus.", 0.50, 70, 14, 1100, 800),
            ("aastocks", "歐美減息週期開啟，銀行股淨息差面臨收窄壓力，0005高位有獲利回吐。", -0.45, 80, 12, 1300, 500),
        ],
        "3690.HK": [
            ("lihkg", "美團KeeTa喺沙特市場擴張迅猛，中東外賣份額直線飆升！美團國際化大有可為！🔥", 0.82, 380, 65, 5100, 14000),
            ("futu", "美團到店業務守住陣地，抖音補貼力度減弱，美團本地生活護城河依然穩固。", 0.62, 230, 35, 3400, 7600),
            ("telegram", "Meituan 3690 call delta buying observed today. Volume ratio 2.4x 30-day average.", 0.70, 115, 28, 1750, 1500),
            ("aastocks", "外賣騎手社保成本預期增加，市場憂慮單均盈利受損，短線股價受壓。", -0.55, 120, 19, 1900, 880),
        ],
        "9618.HK": [
            ("futu", "京東百億補貼見效，用戶活躍度持續提升，物流供應鏈優勢無可替代。", 0.65, 190, 27, 2800, 6300),
            ("lihkg", "京東宣佈新一輪股東分紅方案，現金流充沛，估值被嚴重低估！", 0.70, 220, 34, 3200, 7100),
            ("telegram", "JD retail consumer sentiment index improved slightly this week. Supply chain margins steady.", 0.40, 60, 11, 950, 620),
            ("aastocks", "電子產品消費降級，3C品類換機週期拉長，京東短期營收增速受限。", -0.42, 75, 10, 1150, 490),
        ]
    }

    KR_SOCIAL_TEMPLATES = {
        "005930.KS": [
            ("naver_finance", "삼성전자 드디어 HBM3E 엔비디아 퀄테스트 최종 승인 임박! 8만전자 가자! 🚀", 0.88, 480, 95, 6500, 18000),
            ("daum_finance", "외국인 삼전 연속 순매수 전환. 반도체 바닥 확인하고 턴어라운드 본격화.", 0.72, 260, 42, 3900, 8200),
            ("blind", "DS부문 임직원 성과급 기대감 상승 중. 수율 개선 눈에 띄게 좋아졌음.", 0.65, 190, 31, 2800, 9500),
            ("kakaotalk_telegram", "Samsung Elec institutional buy basket alert. 005930 target price upgraded by foreign brokers.", 0.75, 140, 35, 2100, 3200),
            ("naver_finance", "파운드리 적자 지속 및 레거시 D램 재고 부담 여전함. 추격 매수 금지.", -0.50, 180, 29, 2900, 4100),
        ],
        "000660.KS": [
            ("naver_finance", "SK하이닉스 HBM 공급 독점적 지위 확고. 내년 실적 컨센서스 대폭 상향!", 0.85, 390, 78, 5400, 15000),
            ("daum_finance", "하이닉스 20만원 돌파 카운트다운! AI 반도체 최대 수혜주는 오직 000660.", 0.80, 280, 46, 4100, 9000),
            ("blind", "메모리 사업부 주 7일 가동 수준으로 생산 풀가동 중. 분위기 최고조.", 0.70, 175, 28, 2600, 7800),
            ("naver_finance", "단기 급등에 따른 차익 실현 매물 출회 가능성. 볼린저 상단 저항 주의.", -0.45, 110, 18, 1800, 3300),
        ],
        "035420.KS": [
            ("naver_finance", "네이버 웹툰 나스닥 상장 시너지와 클로바X B2B 매출 가시화! 저평가 매력 부각.", 0.75, 230, 36, 3300, 7400),
            ("daum_finance", "숏폼 숏클립 및 커머스 거래액 반등 성공. 외국계 펀드 유입 지속.", 0.60, 150, 22, 2200, 5100),
            ("blind", "광고 집행 단가 안정화. AI 검색 큐(Cue:) 이용자 체류시간 증가 확인.", 0.58, 120, 18, 1900, 4200),
            ("naver_finance", "국내 내수 플랫폼 규제 리스크 여전. 기술주 전반 밸류에이션 부담.", -0.40, 90, 14, 1400, 2800),
        ],
        "035720.KS": [
            ("naver_finance", "카카오톡 탭 개편 및 메시징 광고 단가 인상으로 분기 흑자폭 확대 기대.", 0.65, 180, 25, 2600, 5800),
            ("daum_finance", "카카오페이·뱅크 금융 계열사 실적 개선 본격화. 턴어라운드 기대.", 0.55, 130, 19, 1950, 4300),
            ("blind", "쇄신 작업 진행 중이나 사법 리스크 완전 해소까지는 시간 소요될 듯.", -0.35, 140, 21, 2100, 5200),
            ("kakaotalk_telegram", "Kakao technical rebound attempt near 45,000 KRW key support level.", 0.40, 75, 12, 1200, 1100),
        ],
        "005380.KS": [
            ("naver_finance", "현대차 인도법인 성공적 상장 및 사상 최대 주주환원 밸류업 정책 발표!", 0.86, 420, 84, 5900, 16000),
            ("daum_finance", "하이브리드 및 제네시스 고수익 차종 비중 확대로 영업이익률 글로벌 최고 수준 유지.", 0.76, 250, 38, 3600, 8100),
            ("blind", "북미 조지아 메타플랜트 조기 가동. 전기차+HEV 유연 생산 체제 완비.", 0.68, 160, 25, 2400, 6900),
            ("naver_finance", "트럼프 관세 우려 및 원화 강세 전환 시 수출 마진 축소 경계감.", -0.48, 130, 20, 2000, 3500),
        ]
    }

    HK_NEWS_TEMPLATES = {
        "0700.HK": [
            ("Tencent Reports Robust Cloud and Gaming Revenue Surge Amid AI Deployment", "Tencent Holdings announced quarterly results exceeding analysts' expectations, driven by solid international game expansion and AI-accelerated advertising tools.", 0.80, "South China Morning Post"),
            ("Hong Kong Tech Equities Gain as Southbound Capital Inflows Hit Multi-Month High", "Tencent and key Hang Seng Tech constituents saw substantial institutional buying through the Stock Connect channel.", 0.65, "Bloomberg News"),
            ("Regulatory Landscape Remains Stable for Major Chinese Internet Platforms", "China's digital economy oversight focuses on high-quality development, fostering a favorable backdrop for Tencent's core business lines.", 0.55, "Reuters"),
        ],
        "9988.HK": [
            ("Alibaba Cloud Announces Broad Price Reductions to Accelerate Enterprise AI Adoption", "Alibaba Group expanded its AI infrastructure capabilities, signaling stronger enterprise cloud adoption across APAC.", 0.68, "Financial Times"),
            ("E-Commerce Competition Stabilizes in Domestic Market as Taobao Focuses on Merchant ROI", "Industry checks indicate stabilizing take rates and improved user retention for Alibaba's retail marketplaces.", 0.52, "Caixin Global"),
            ("Alibaba Repurchases $4.8 Billion in Shares Under Expanded Capital Return Program", "Management reiterated its commitment to shareholder value through aggressive share retirements.", 0.74, "Wall Street Journal"),
        ],
        "0005.HK": [
            ("HSBC Holdings Unveils $3 Billion Share Buyback Following Strong Wealth Management Profit", "HSBC Holdings plc posted resilient net interest income and double-digit growth in Asia wealth fees.", 0.78, "Financial Times"),
            ("Hong Kong Banking Sector Benefits from High Base Rate Environment", "Leading lenders maintain solid net interest margins while provisions for commercial real estate remain contained.", 0.50, "HKET"),
        ],
        "3690.HK": [
            ("Meituan Expands KeeTa Food Delivery Network Across Middle East Markets", "Meituan's international brand KeeTa secured strong initial transaction volumes in Riyadh, demonstrating scalable overseas expansion.", 0.81, "South China Morning Post"),
            ("Local Life Service Margins Expand as Competitive Subsidies Normalize", "Meituan demonstrated superior merchant network effects, consolidating its leadership position in on-demand delivery.", 0.64, "Bloomberg News"),
        ],
        "9618.HK": [
            ("JD.com Outlines Supply Chain Automation Upgrades and Higher Shareholder Dividends", "JD.com reaffirmed its supply chain cost efficiency and announced an enhanced capital allocation framework.", 0.67, "Reuters"),
            ("Consumer Electronics Demand Shows Early Signs of Recovery in Tier-1 Cities", "Retail data highlighted stable demand for appliances and mobile devices ahead of seasonal promotion cycles.", 0.48, "Xinhua Finance"),
        ]
    }

    KR_NEWS_TEMPLATES = {
        "005930.KS": [
            ("Samsung Electronics Prepares Next-Gen HBM3E Memory Chips for Leading AI Accelerator Platforms", "Industry sources confirm validation progress for Samsung's advanced 12-layer HBM3E memory with premier global customers.", 0.84, "Korea Economic Daily"),
            ("Foreign Institutional Inflows Into KOSPI Tech Heavyweights Accelerate", "Samsung Electronics and domestic semiconductor makers led institutional basket purchases in Seoul trading.", 0.68, "Maeil Business"),
        ],
        "000660.KS": [
            ("SK Hynix Expands High-Bandwidth Memory Dominance with Next-Generation Production Capacity", "SK Hynix reiterated its sold-out status for near-term HBM supply amid unprecedented hyperscaler demand.", 0.86, "Reuters"),
            ("Semiconductor Super-Cycle Continues Driven by Sovereign AI and Datacenter Capex", "Analysts raise target prices for SK Hynix, citing pricing power in premium enterprise SSDs and HBM.", 0.75, "Yonhap News"),
        ],
        "035420.KS": [
            ("NAVER Accelerates Enterprise B2B Generative AI Deployments Across Financial Services", "NAVER Cloud signed landmark AI infrastructure agreements with major domestic financial groups.", 0.71, "Korea Herald"),
            ("Webtoon Entertainment Expansion Enhances NAVER's Global Digital Content Footprint", "Global IP monetization and advertising momentum drive positive earnings revisions for NAVER.", 0.60, "The Bell"),
        ],
        "035720.KS": [
            ("Kakao Focuses on AI-Powered Talk Channels to Boost Advertising Average Revenue Per User", "Kakao unveiled new conversational AI features integrated directly into its ubiquitous messenger application.", 0.62, "Business Korea"),
        ],
        "005380.KS": [
            ("Hyundai Motor Group Unveils Historic Shareholder Return Framework and India IPO Milestones", "Hyundai Motor confirmed multi-year dividend growth targets and continued market share gains in hybrid vehicles.", 0.85, "Bloomberg"),
            ("North American Hybrid Vehicle Sales Drive Record Operating Margins for Hyundai Motor", "Favorable product mix and resilient consumer demand support elevated average selling prices.", 0.73, "Automotive News"),
        ]
    }

    @classmethod
    def generate_social_posts(
        cls,
        ticker: str,
        asset_name: str,
        days_back: int = 10,
        exchange: str = "hkex"
    ) -> List[SocialPost]:
        """Generates realistic timestamped social posts with authentic engagement metrics."""
        posts: List[SocialPost] = []
        is_krx = exchange.lower() == "krx" or ticker.endswith(".KS")
        template_dict = cls.KR_SOCIAL_TEMPLATES if is_krx else cls.HK_SOCIAL_TEMPLATES
        templates = template_dict.get(ticker, list(template_dict.values())[0])

        now = datetime.now(timezone.utc)
        np.random.seed(abs(hash(ticker + "social")) % (2**32))

        # Generate 4-8 posts per day across the lookback period
        for day in range(days_back, -1, -1):
            day_date = now - timedelta(days=day)
            num_posts_today = np.random.randint(4, 9)
            
            for p_idx in range(num_posts_today):
                tpl = templates[np.random.choice(len(templates))]
                platform, text, base_sentiment, base_likes, base_reposts, base_views, base_followers = tpl

                # Add intraday timestamp variation
                hour = np.random.randint(1, 15)
                minute = np.random.randint(0, 60)
                post_time = day_date.replace(hour=hour, minute=minute, second=np.random.randint(0, 60))

                # Perturb engagement numbers realistically
                engagement_multiplier = np.random.uniform(0.6, 1.8)
                likes = int(base_likes * engagement_multiplier)
                reposts = int(base_reposts * engagement_multiplier)
                views = int(base_views * engagement_multiplier)
                followers = int(base_followers * np.random.uniform(0.7, 1.5))

                # Slight text variations
                post_id = hashlib.md5(f"{ticker}_{post_time}_{p_idx}_{text[:10]}".encode()).hexdigest()[:12]
                
                posts.append(
                    SocialPost(
                        post_id=post_id,
                        ticker=ticker,
                        platform=platform,
                        timestamp=post_time,
                        text=text,
                        author=f"trader_{np.random.randint(100, 999)}",
                        author_followers=followers,
                        likes=likes,
                        reposts=reposts,
                        views=views,
                        url=f"https://community.example.com/post/{post_id}",
                        raw_metadata={"synthetic": True, "base_sentiment": base_sentiment}
                    )
                )

        logger.info(f"Synthesized {len(posts)} social posts for {ticker} over {days_back} days.")
        return sorted(posts, key=lambda x: x.timestamp)

    @classmethod
    def generate_news_articles(
        cls,
        ticker: str,
        asset_name: str,
        days_back: int = 10,
        exchange: str = "hkex"
    ) -> List[NewsArticle]:
        """Generates realistic timestamped news articles matching Exa schema."""
        articles: List[NewsArticle] = []
        is_krx = exchange.lower() == "krx" or ticker.endswith(".KS")
        news_dict = cls.KR_NEWS_TEMPLATES if is_krx else cls.HK_NEWS_TEMPLATES
        templates = news_dict.get(ticker, list(news_dict.values())[0])

        now = datetime.now(timezone.utc)
        np.random.seed(abs(hash(ticker + "news")) % (2**32))

        # Generate 1-3 news articles every 2-3 days
        for day in range(days_back, -1, -2):
            day_date = now - timedelta(days=day)
            num_articles = np.random.randint(1, 3)

            for a_idx in range(num_articles):
                tpl = templates[np.random.choice(len(templates))]
                title, snippet, sentiment, publisher = tpl

                hour = np.random.randint(0, 10) # news releases often before market open
                minute = np.random.randint(0, 60)
                art_time = day_date.replace(hour=hour, minute=minute, second=0)

                art_id = hashlib.md5(f"{ticker}_{art_time}_{title}".encode()).hexdigest()[:12]
                articles.append(
                    NewsArticle(
                        article_id=art_id,
                        ticker=ticker,
                        source_engine="exa",
                        timestamp=art_time,
                        title=title,
                        snippet=snippet,
                        url=f"https://www.financialnews.com/article/{art_id}",
                        publisher=publisher,
                        relevance_score=round(float(np.random.uniform(0.85, 0.99)), 3)
                    )
                )

        logger.info(f"Synthesized {len(articles)} news articles for {ticker} over {days_back} days.")
        return sorted(articles, key=lambda x: x.timestamp)

