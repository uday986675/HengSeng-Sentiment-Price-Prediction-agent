"""Command-Line Interface (CLI) entrypoint for HengSeng Sentiment Price Prediction Agent."""
import argparse
import sys
from pathlib import Path
from agent.config.settings import get_default_config, PipelineConfig
from agent.ingestion.pipeline import IngestionPipeline
from agent.features.feature_engineering import FeatureEngineer
from agent.modeling.backtest import BacktestEngine
from agent.reporting.dashboard import VisualReportGenerator
from agent.utils.logger import setup_logger

logger = setup_logger("CLI")


def run_pipeline(
    exchange: str = "hkex",
    force_refresh: bool = False,
    model_type: str = "ensemble"
):
    """Executes the complete end-to-end pipeline: Ingestion -> Features -> Model & Validation -> Visual Report."""
    logger.info(f"Starting HengSeng Sentiment Price Prediction Agent for exchange: {exchange.upper()}")
    
    # 1. Load Config
    config = get_default_config(exchange=exchange.lower())
    config.model.model_type = model_type
    
    # 2. Ingestion
    ingestion = IngestionPipeline(config=config)
    social_df, news_df, prices_df = ingestion.run(force_refresh=force_refresh)

    # 3. Feature Engineering & Per-Ticker Extraction
    logger.info("Computing creative sentiment features, velocity, z-scores, and technical levels...")
    engineer = FeatureEngineer()
    asset_meta = {a.ticker: a.name for a in config.exchanges[config.exchange].assets}
    currency = config.exchanges[config.exchange].currency

    ml_df, summaries = engineer.process_all_tickers(
        social_df=social_df,
        news_df=news_df,
        prices_df=prices_df,
        asset_metadata=asset_meta,
        currency=currency
    )

    if ml_df.empty:
        logger.error("No valid features could be computed. Check data ingestion.")
        return

    # Print Per-Ticker Extraction Summary
    print("\n" + "=" * 80)
    print(f"  PER-TICKER EXTRACTION & SIGNAL SUMMARY ({config.exchange.upper()})")
    print("=" * 80)
    for s in summaries:
        print(f"[{s.ticker}] {s.asset_name}")
        print(f"  Current Price:       {s.current_price:.2f} {s.currency}")
        print(f"  Sentiment:           {s.aggregate_sentiment_score:+.2f} ({s.sentiment_direction.upper()})")
        print(f"  Social Mentions:     {s.total_social_mentions_10d} (Z-Score: {s.mention_volume_zscore:+.2f}σ)")
        print(f"  Sentiment Velocity:  {s.sentiment_velocity:+.3f} | Momentum: {s.sentiment_momentum:+.3f}")
        print(f"  News vs Social Div:  {s.news_social_divergence:+.3f}")
        print(f"  Suggested Position:  {s.suggested_position}")
        print(f"  Rationale:           {s.position_rationale}")
        print(f"  Key Levels:          Pivot: {s.support_resistance.get('pivot')} | S1: {s.support_resistance.get('support_1')} | R1: {s.support_resistance.get('resistance_1')}")
        print("-" * 80)

    # 4. Out-of-Sample Walk-Forward Validation & Backtesting
    backtester = BacktestEngine(n_splits=config.model.cv_n_splits, test_size=config.model.test_size_days, random_state=config.model.random_state)
    backtest_result = backtester.run_validation(ml_df=ml_df, exchange=config.exchange)
    backtest_result.print_summary()

    # 5. Generate Visual Report Dashboard
    reporter = VisualReportGenerator(output_dir=config.reports_dir)
    report_path = reporter.generate_dashboard(
        summaries=summaries,
        ml_df=ml_df,
        backtest_result=backtest_result,
        exchange=config.exchange
    )
    print(f"\n🎉 Visual Report generated successfully! Open in browser: {report_path.resolve()}\n")


def main():
    """Main CLI entrypoint parser."""
    parser = argparse.ArgumentParser(
        description="HengSeng Sentiment Price Prediction Agent - Quantitative Alternative Data Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run the full end-to-end pipeline")
    run_parser.add_argument(
        "--exchange",
        type=str,
        default="hkex",
        choices=["hkex", "krx"],
        help="Target exchange to process"
    )
    run_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and force re-scraping / re-fetching"
    )
    run_parser.add_argument(
        "--model-type",
        type=str,
        default="ensemble",
        choices=["ensemble", "ridge", "random_forest", "gradient_boosting"],
        help="Model architecture for next-bar return/price prediction"
    )

    # Command: backtest
    bt_parser = subparsers.add_parser("backtest", help="Run walk-forward validation and backtest")
    bt_parser.add_argument("--exchange", type=str, default="hkex", choices=["hkex", "krx"])

    # Command: report
    rep_parser = subparsers.add_parser("report", help="Regenerate visual report dashboard from processed data")
    rep_parser.add_argument("--exchange", type=str, default="hkex", choices=["hkex", "krx"])

    args = parser.parse_args()

    if args.command == "run" or args.command is None:
        exchange = getattr(args, "exchange", "hkex")
        force = getattr(args, "force_refresh", False)
        model = getattr(args, "model_type", "ensemble")
        run_pipeline(exchange=exchange, force_refresh=force, model_type=model)
    elif args.command == "backtest":
        run_pipeline(exchange=args.exchange, force_refresh=False)
    elif args.command == "report":
        run_pipeline(exchange=args.exchange, force_refresh=False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
