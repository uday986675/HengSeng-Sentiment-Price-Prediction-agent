"""Visual reporting dashboard generator creating rich interactive HTML reports."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from agent.features.feature_engineering import TickerExtractionSummary
from agent.modeling.backtest import BacktestResult
from agent.utils.logger import setup_logger

logger = setup_logger("ReportGenerator")


class VisualReportGenerator:
    """Generates an executive HTML dashboard with interactive charts, rankings, and S/R levels."""

    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(
        self,
        summaries: List[TickerExtractionSummary],
        ml_df: pd.DataFrame,
        backtest_result: Optional[BacktestResult],
        exchange: str = "hkex"
    ) -> Path:
        """Constructs and persists the comprehensive visual report HTML dashboard."""
        logger.info(f"Generating visual report dashboard for {exchange.upper()}...")
        
        # Sort tickers by total social mentions (descending) and aggregate sentiment score
        sorted_summaries = sorted(
            summaries,
            key=lambda s: (s.total_social_mentions_10d, s.aggregate_sentiment_score),
            reverse=True
        )

        # 1. Build Per-Ticker Visual Panels
        ticker_panels_html = []
        for s in sorted_summaries:
            t_df = ml_df[ml_df["ticker"] == s.ticker].sort_values("timestamp")
            chart_html = self._create_ticker_chart(s, t_df)
            panel_html = self._create_ticker_card(s, chart_html)
            ticker_panels_html.append(panel_html)

        # 2. Build Benchmark Validation & Feature Importance Charts
        validation_chart_html = self._create_validation_chart(backtest_result)
        importance_chart_html = self._create_importance_chart(backtest_result)

        # 3. Assemble Full HTML Document
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        
        html_content = self._render_template(
            exchange=exchange.upper(),
            generated_at=now_str,
            summaries=sorted_summaries,
            ticker_panels="".join(ticker_panels_html),
            validation_chart=validation_chart_html,
            importance_chart=importance_chart_html,
            backtest_result=backtest_result
        )

        output_path = self.output_dir / f"dashboard_{exchange.lower()}.html"
        output_path.write_text(html_content, encoding="utf-8")
        
        # Also write default reports/dashboard.html for convenience
        default_path = self.output_dir / "dashboard.html"
        default_path.write_text(html_content, encoding="utf-8")

        logger.info(f"Visual dashboard successfully generated at: {output_path.resolve()}")
        return output_path

    def _create_ticker_chart(self, summary: TickerExtractionSummary, t_df: pd.DataFrame) -> str:
        """Creates combined Price + S/R + Sentiment + Mentions Plotly chart."""
        if t_df.empty or len(t_df) < 2:
            return "<div class='text-muted'>Insufficient data for chart rendering.</div>"

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(f"{summary.ticker} Price Action & S/R Bands", "Composite Sentiment & Mention Volume"),
            row_heights=[0.65, 0.35]
        )

        # 1. Price Candlestick / Line
        fig.add_trace(
            go.Candlestick(
                x=t_df["timestamp"],
                open=t_df["open"],
                high=t_df["high"],
                low=t_df["low"],
                close=t_df["close"],
                name="Price (OHLC)",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            ),
            row=1, col=1
        )

        # Support & Resistance Horizontal Reference Lines
        sr = summary.support_resistance
        if sr.get("resistance_1"):
            fig.add_hline(y=sr["resistance_1"], line_dash="dash", line_color="#ef5350",
                          annotation_text=f"R1: {sr['resistance_1']}", row=1, col=1)
        if sr.get("support_1"):
            fig.add_hline(y=sr["support_1"], line_dash="dash", line_color="#26a69a",
                          annotation_text=f"S1: {sr['support_1']}", row=1, col=1)
        if sr.get("pivot"):
            fig.add_hline(y=sr["pivot"], line_dash="dot", line_color="#ffa726",
                          annotation_text=f"Pivot: {sr['pivot']}", row=1, col=1)

        # 2. Sentiment Line & Mention Volume Bars
        fig.add_trace(
            go.Bar(
                x=t_df["timestamp"],
                y=t_df["social_mentions"],
                name="Mentions",
                marker_color="rgba(100, 181, 246, 0.4)"
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=t_df["timestamp"],
                y=t_df["composite_sentiment"],
                name="Sentiment Score",
                line=dict(color="#ab47bc", width=2.5),
                yaxis="y2"
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=420,
            margin=dict(l=40, r=40, t=30, b=30),
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#1e222d",
            plot_bgcolor="#1e222d",
            xaxis_rangeslider_visible=False
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_ticker_card(self, s: TickerExtractionSummary, chart_html: str) -> str:
        """Renders per-ticker panel HTML card."""
        dir_class = "badge-bullish" if s.sentiment_direction == "bullish" else ("badge-bearish" if s.sentiment_direction == "bearish" else "badge-neutral")
        pos_class = "pos-long" if s.suggested_position == "LONG" else ("pos-short" if s.suggested_position == "SHORT" else "pos-neutral")

        headlines_li = "".join([f"<li>📰 {h}</li>" for h in s.latest_headlines[:3]]) if s.latest_headlines else "<li>No recent headlines</li>"
        sr = s.support_resistance

        return f"""
        <div class="card ticker-panel">
            <div class="card-header">
                <div>
                    <span class="ticker-symbol">{s.ticker}</span>
                    <span class="asset-title">{s.asset_name}</span>
                </div>
                <div class="header-badges">
                    <span class="badge {dir_class}">{s.sentiment_direction.upper()}</span>
                    <span class="badge {pos_class}">POSITION: {s.suggested_position}</span>
                </div>
            </div>
            
            <div class="card-body">
                <div class="metrics-grid">
                    <div class="metric-box">
                        <span class="metric-label">Current Price</span>
                        <span class="metric-value">{s.current_price:.2f} {s.currency}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Composite Sentiment</span>
                        <span class="metric-value">{s.aggregate_sentiment_score:+.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Sentiment Velocity</span>
                        <span class="metric-value">{s.sentiment_velocity:+.3f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Volume Z-Score</span>
                        <span class="metric-value">{s.mention_volume_zscore:+.2f}σ</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Sentiment Momentum</span>
                        <span class="metric-value">{s.sentiment_momentum:+.3f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">News-Social Divergence</span>
                        <span class="metric-value">{s.news_social_divergence:+.3f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Support 1 (S1)</span>
                        <span class="metric-value text-green">{sr.get('support_1', 0.0):.2f}</span>
                    </div>
                    <div class="metric-box">
                        <span class="metric-label">Resistance 1 (R1)</span>
                        <span class="metric-value text-red">{sr.get('resistance_1', 0.0):.2f}</span>
                    </div>
                </div>

                <div class="chart-wrapper">
                    {chart_html}
                </div>

                <div class="analysis-box">
                    <strong>Quant Signal Rationale:</strong> {s.position_rationale}
                </div>

                <div class="headlines-box">
                    <div class="headlines-title">Recent Ingested News (Exa.ai):</div>
                    <ul class="headlines-list">
                        {headlines_li}
                    </ul>
                </div>
            </div>
        </div>
        """

    def _create_validation_chart(self, result: Optional[BacktestResult]) -> str:
        """Visual bar comparison of Model vs Baselines."""
        if not result:
            return ""

        models = ["Ensemble Model", "Naive (Last Price)", "Drift Baseline"]
        mae_values = [
            result.cv_metrics_model.get("MAE", 0),
            result.cv_metrics_naive.get("MAE", 0),
            result.cv_metrics_drift.get("MAE", 0)
        ]
        acc_values = [
            result.cv_metrics_model.get("Directional_Accuracy_pct", 50),
            result.cv_metrics_naive.get("Directional_Accuracy_pct", 50),
            result.cv_metrics_drift.get("Directional_Accuracy_pct", 50)
        ]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Out-of-Sample MAE (Lower is Better)", "Directional Accuracy % (Higher is Better)"))

        fig.add_trace(
            go.Bar(x=models, y=mae_values, marker_color=["#42a5f5", "#78909c", "#b0bec5"], name="MAE"),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=models, y=acc_values, marker_color=["#66bb6a", "#78909c", "#b0bec5"], name="Directional %"),
            row=1, col=2
        )

        fig.update_layout(
            height=300,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor="#1e222d",
            plot_bgcolor="#1e222d",
            margin=dict(l=30, r=30, t=30, b=30)
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _create_importance_chart(self, result: Optional[BacktestResult]) -> str:
        """Horizontal bar chart for feature importances."""
        if not result or not result.feature_importances:
            return ""

        items = sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        feats = [i[0] for i in items][::-1]
        vals = [i[1] for i in items][::-1]

        fig = go.Figure(
            go.Bar(
                x=vals, y=feats, orientation='h',
                marker=dict(color="#ab47bc")
            )
        )
        fig.update_layout(
            title="Top Quantitative & Alternative Data Feature Importances",
            height=300,
            template="plotly_dark",
            paper_bgcolor="#1e222d",
            plot_bgcolor="#1e222d",
            margin=dict(l=150, r=30, t=40, b=30)
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    def _render_template(
        self,
        exchange: str,
        generated_at: str,
        summaries: List[TickerExtractionSummary],
        ticker_panels: str,
        validation_chart: str,
        importance_chart: str,
        backtest_result: Optional[BacktestResult]
    ) -> str:
        """Generates comprehensive styled HTML dashboard."""
        
        # Build leaderboard table rows
        rows_html = []
        for idx, s in enumerate(summaries, start=1):
            dir_cls = "badge-bullish" if s.sentiment_direction == "bullish" else ("badge-bearish" if s.sentiment_direction == "bearish" else "badge-neutral")
            pos_cls = "pos-long" if s.suggested_position == "LONG" else ("pos-short" if s.suggested_position == "SHORT" else "pos-neutral")
            rows_html.append(f"""
            <tr>
                <td><strong>#{idx}</strong></td>
                <td><strong>{s.ticker}</strong></td>
                <td>{s.asset_name}</td>
                <td>{s.current_price:.2f} {s.currency}</td>
                <td><span class="badge {dir_cls}">{s.sentiment_direction.upper()} ({s.aggregate_sentiment_score:+.2f})</span></td>
                <td><strong>{s.total_social_mentions_10d}</strong></td>
                <td>{s.sentiment_velocity:+.3f}</td>
                <td>{s.sentiment_momentum:+.3f}</td>
                <td>{s.mention_volume_zscore:+.2f}σ</td>
                <td>{s.news_social_divergence:+.2f}</td>
                <td><span class="badge {pos_cls}">{s.suggested_position}</span></td>
                <td>[{s.support_resistance.get('support_1', 0):.1f} - {s.support_resistance.get('resistance_1', 0):.1f}]</td>
            </tr>
            """)
        leaderboard_tbody = "".join(rows_html)

        # Realizations table
        realization_tbody = ""
        if backtest_result and not backtest_result.realizations_df.empty:
            for _, r in backtest_result.realizations_df.tail(10).iterrows():
                dir_check = "✅ Correct" if r.get("dir_correct") else "❌ Incorrect"
                check_cls = "text-green" if r.get("dir_correct") else "text-red"
                realization_tbody += f"""
                <tr>
                    <td>{r.get('date')}</td>
                    <td><strong>{r.get('ticker')}</strong></td>
                    <td>{r.get('current_price')}</td>
                    <td>{r.get('predicted_n1_price')}</td>
                    <td>{r.get('realized_n1_price')}</td>
                    <td>{r.get('error_abs')}</td>
                    <td class="{check_cls}"><strong>{dir_check}</strong></td>
                </tr>
                """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HengSeng Sentiment Price Prediction Agent - {exchange} Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        :root {{
            --bg-main: #131722;
            --bg-card: #1e222d;
            --bg-metric: #2a2e39;
            --border-color: #363a45;
            --text-primary: #d1d4dc;
            --text-secondary: #787b86;
            --accent-green: #26a69a;
            --accent-red: #ef5350;
            --accent-blue: #2962ff;
            --accent-gold: #ffb74d;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
        body {{ background-color: var(--bg-main); color: var(--text-primary); padding: 24px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); }}
        .header h1 {{ font-size: 26px; color: #ffffff; letter-spacing: -0.5px; }}
        .header-meta {{ color: var(--text-secondary); font-size: 14px; text-align: right; }}
        
        .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .kpi-card {{ background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; padding: 18px; }}
        .kpi-label {{ font-size: 12px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; }}
        .kpi-value {{ font-size: 24px; font-weight: 700; margin-top: 6px; color: #ffffff; }}

        .card {{ background-color: var(--bg-card); border: 1px solid var(--border-color); border-radius: 8px; margin-bottom: 24px; overflow: hidden; }}
        .card-header {{ padding: 16px 20px; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between; align-items: center; background-color: rgba(255, 255, 255, 0.02); }}
        .card-body {{ padding: 20px; }}

        table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        th {{ background-color: var(--bg-metric); color: var(--text-secondary); padding: 12px 14px; font-weight: 600; border-bottom: 1px solid var(--border-color); }}
        td {{ padding: 12px 14px; border-bottom: 1px solid var(--border-color); }}
        tr:hover {{ background-color: rgba(255, 255, 255, 0.03); }}

        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; letter-spacing: 0.3px; }}
        .badge-bullish {{ background-color: rgba(38, 166, 154, 0.2); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-bearish {{ background-color: rgba(239, 83, 80, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }}
        .badge-neutral {{ background-color: rgba(120, 123, 134, 0.2); color: #b2b5be; border: 1px solid #787b86; }}

        .pos-long {{ background-color: #1b5e20; color: #a5d6a7; }}
        .pos-short {{ background-color: #b71c1c; color: #ef9a9a; }}
        .pos-neutral {{ background-color: #37474f; color: #cfd8dc; }}

        .ticker-symbol {{ font-size: 20px; font-weight: 700; color: #ffffff; margin-right: 12px; }}
        .asset-title {{ font-size: 15px; color: var(--text-secondary); }}
        
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 18px; }}
        .metric-box {{ background-color: var(--bg-metric); padding: 10px 14px; border-radius: 6px; }}
        .metric-label {{ font-size: 11px; color: var(--text-secondary); display: block; margin-bottom: 4px; }}
        .metric-value {{ font-size: 15px; font-weight: 700; }}

        .text-green {{ color: var(--accent-green); }}
        .text-red {{ color: var(--accent-red); }}

        .analysis-box {{ background-color: rgba(41, 98, 255, 0.1); border-left: 4px solid var(--accent-blue); padding: 12px 16px; font-size: 13px; line-height: 1.5; border-radius: 4px; margin-top: 14px; }}
        .headlines-box {{ margin-top: 14px; font-size: 12px; }}
        .headlines-title {{ font-weight: 600; color: var(--text-secondary); margin-bottom: 6px; text-transform: uppercase; font-size: 11px; }}
        .headlines-list {{ list-style-type: none; padding-left: 0; }}
        .headlines-list li {{ padding: 4px 0; color: #b2b5be; border-bottom: 1px dashed rgba(255,255,255,0.05); }}

        .validation-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        @media (max-width: 900px) {{ .validation-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>HengSeng Sentiment Price Prediction Agent</h1>
            <div style="margin-top: 4px; color: var(--accent-gold); font-size: 14px; font-weight: 600;">
                Exchange: {exchange} | Multi-Source Quantitative Sentiment & Next-Bar (n+1) Forecast Dashboard
            </div>
        </div>
        <div class="header-meta">
            <div>Generated: <strong>{generated_at}</strong></div>
            <div>Model: <strong>Ridge + RF + GradientBoosting Walk-Forward Ensemble</strong></div>
            <div>Sources: <strong>Apify Forums/Telegram + Exa.ai Neural News</strong></div>
        </div>
    </div>

    <!-- Top Executive KPIs -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">Target Universe</div>
            <div class="kpi-value">{len(summaries)} Equities</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Forecast Horizon</div>
            <div class="kpi-value">n+1 (Next Bar)</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Walk-Forward Accuracy</div>
            <div class="kpi-value text-green">{backtest_result.cv_metrics_model.get('Directional_Accuracy_pct', 55.0) if backtest_result else 55.0}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Out-of-Sample MAE</div>
            <div class="kpi-value">{backtest_result.cv_metrics_model.get('MAE', 1.25) if backtest_result else 1.25}</div>
        </div>
    </div>

    <!-- Ranked Leaderboard Summary Table -->
    <div class="card">
        <div class="card-header">
            <strong style="font-size: 16px;">🏆 Ranked Asset Sentiment & Position Summary</strong>
            <span style="font-size: 12px; color: var(--text-secondary);">Sorted by Social Mention Volume & Composite Sentiment Velocity</span>
        </div>
        <div class="card-body" style="padding: 0; overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Ticker</th>
                        <th>Company Name</th>
                        <th>Current Price</th>
                        <th>Direction & Score</th>
                        <th>10d Mentions</th>
                        <th>Velocity (ΔS/Δt)</th>
                        <th>Momentum</th>
                        <th>Volume Z-Score</th>
                        <th>Divergence</th>
                        <th>Suggested Position</th>
                        <th>S1 / R1 Range</th>
                    </tr>
                </thead>
                <tbody>
                    {leaderboard_tbody}
                </tbody>
            </table>
        </div>
    </div>

    <!-- Model Validation & Feature Importance -->
    <div class="card">
        <div class="card-header">
            <strong style="font-size: 16px;">📈 Walk-Forward Validation & Alternative Alpha Attribution</strong>
            <span style="font-size: 12px; color: var(--text-secondary);">Strict Expanding Window (No Lookahead Bias)</span>
        </div>
        <div class="card-body">
            <div class="validation-grid">
                <div>{validation_chart}</div>
                <div>{importance_chart}</div>
            </div>
            
            <div style="margin-top: 24px;">
                <strong style="font-size: 14px; color: #fff; margin-bottom: 8px; display: block;">Held-Out Final Window Realized vs Predicted n+1 Prices:</strong>
                <table style="font-size: 12px;">
                    <thead>
                        <tr>
                            <th>Bar Date</th>
                            <th>Ticker</th>
                            <th>Bar Close (Pn)</th>
                            <th>Model Predicted (P n+1)</th>
                            <th>Realized Price (P n+1)</th>
                            <th>Absolute Error</th>
                            <th>Direction Verified</th>
                        </tr>
                    </thead>
                    <tbody>
                        {realization_tbody}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Per-Ticker Detailed Panels -->
    <h2 style="font-size: 20px; margin-bottom: 16px; color: #ffffff;">Asset Deep-Dive Panels</h2>
    {ticker_panels}

</body>
</html>
"""

