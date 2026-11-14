"""Financial sentiment scoring engine supporting multilingual financial text."""
import re
from typing import Dict, Tuple


class FinancialSentimentAnalyzer:
    """
    Multilingual financial sentiment scoring engine.
    Calibrated specifically for Chinese (HKEX forums/LIHKG), Korean (KRX/Naver), and English market commentary.
    """

    # Positive financial keywords
    POS_EN = [
        "surge", "rally", "gain", "bullish", "beat", "outperform", "buyback", "dividend",
        "upgrade", "breakout", "rebound", "growth", "strong", "profit", "expansion",
        "approval", "catalyst", "record", "optimistic", "inflow", "undervalued", "upside"
    ]
    NEG_EN = [
        "plunge", "drop", "slump", "bearish", "miss", "underperform", "downgrade", "selloff",
        "loss", "decline", "warning", "weak", "cut", "probe", "investigation", "tariff",
        "penalty", "default", "overvalued", "outflow", "short", "downside", "headwind"
    ]

    POS_ZH = [
        "升", "漲", "涨", "大買", "大买", "飆升", "飙升", "突破", "買入", "买入", "回購", "回购",
        "超預期", "超预期", "多頭", "多头", "反彈", "反弹", "走強", "走强", "增持", "底背馳", "底背离",
        "賺", "赚", "看好", "企穩", "企稳", "低估", "創新高", "创新高", "護城河", "护城河"
    ]
    NEG_ZH = [
        "跌", "沽空", "散貨", "散货", "暴跌", "套牢", "減持", "减持", "虧損", "亏损", "破位",
        "空頭", "空头", "走弱", "割肉", "暴雷", "監管", "监管", "死雀", "看空", "獲利回吐", "获利回吐",
        "受壓", "受压", "下行", "陷阱", "清倉", "清仓"
    ]

    POS_KR = [
        "상승", "급등", "매수", "호재", "돌파", "반등", "흑자", "턴어라운드", "주주환원",
        "승인", "신고가", "우상향", "수익", "성장", "순매수", "독점", "상향", "저평가", "기대"
    ]
    NEG_KR = [
        "하락", "급락", "매도", "악재", "적자", "손실", "이탈", "경계", "차익실현", "하향",
        "우려", "리스크", "규제", "과열", "부담", "순매도", "축소", "폭락", "투매"
    ]

    # Intensifiers
    INTENSIFIERS = {
        "very": 1.4, "extremely": 1.8, "strongly": 1.5, "hugely": 1.6,
        "大": 1.3, "超": 1.5, "極": 1.6, "極度": 1.8,
        "매우": 1.4, "대폭": 1.6, "극대": 1.7, "완전": 1.5
    }

    # Negators (matched with word boundary or specific prefixes)
    NEGATORS_EN = [r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bunlikely\b", r"\bhardly\b", r"\bwithout\b"]
    NEGATORS_CJK = ["不是", "没有", "並非", "并非", "并未", "並未", "不看好", "아닌", "아니다", "없다", "없음", "못함"]

    def score_text(self, text: str) -> Tuple[float, str]:
        """
        Analyzes text and returns (sentiment_score, direction).
        Score is continuous in [-1.0, 1.0].
        Direction is 'bullish', 'bearish', or 'neutral'.
        """
        if not text or not isinstance(text, str):
            return 0.0, "neutral"

        text_lower = text.lower()
        pos_score = 0.0
        neg_score = 0.0

        # English analysis
        for word in self.POS_EN:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                pos_score += 1.0
        for word in self.NEG_EN:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                neg_score += 1.0

        # Chinese analysis
        for token in self.POS_ZH:
            if token in text:
                pos_score += 1.2
        for token in self.NEG_ZH:
            if token in text:
                neg_score += 1.2

        # Korean analysis
        for token in self.POS_KR:
            if token in text:
                pos_score += 1.2
        for token in self.NEG_KR:
            if token in text:
                neg_score += 1.2

        # Check for emojis
        if any(emoji in text for emoji in ["🚀", "🔥", "📈", "💰", "💪", "🎉"]):
            pos_score += 1.5
        if any(emoji in text for emoji in ["📉", "💀", "😭", "🩸", "⚠️", "❌"]):
            neg_score += 1.5

        # Check for negation with proper boundary check for English
        has_negation_en = any(re.search(pat, text_lower) for pat in self.NEGATORS_EN)
        has_negation_cjk = any(cjk in text for cjk in self.NEGATORS_CJK)
        if has_negation_en or has_negation_cjk:
            # Dampen or invert polarity slightly
            pos_score, neg_score = neg_score * 0.7, pos_score * 0.7

        total_matches = pos_score + neg_score
        if total_matches == 0:
            return 0.0, "neutral"

        # Normalized bounded score [-1.0, 1.0]
        raw_score = (pos_score - neg_score) / (total_matches + 1.0)
        score = max(-1.0, min(1.0, raw_score * 1.5))

        if score >= 0.15:
            direction = "bullish"
        elif score <= -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        return round(float(score), 4), direction
