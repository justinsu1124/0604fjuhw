#!/usr/bin/env python3
"""RSRS × Volume/MA Confirmation — NautilusTrader backtest runner.

Usage
-----
    FMP_API_KEY=<key> python backtest.py [--start 2000-01-01] [--end 2026-01-01]

Workflow
--------
1. Fetch SPY daily OHLCV from FMP.
2. Build NautilusTrader BacktestEngine with a simulated venue.
3. Run RSRSStrategy (right-skewed signal + volume/MA filter).
4. Print performance summary and save equity-curve plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from data_loader import fetch_fmp_daily, dataframe_to_bars
from rsrs_strategy import RSRSStrategy, RSRSStrategyConfig

sns.set_theme(style="whitegrid")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BAR_TYPE_STR = "SPY.XNAS-1-DAY-LAST-EXTERNAL"
INSTRUMENT_ID_STR = "SPY.XNAS"


def performance_stats(
    equity: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    n = len(rets)
    if n < 2:
        return {}
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_return = (1 + total_return) ** (periods_per_year / n) - 1
    ann_vol = rets.std() * np.sqrt(periods_per_year)
    sharpe = (ann_return - rf) / ann_vol if ann_vol > 0 else float("nan")
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "Total return": total_return,
        "Annualized return": ann_return,
        "Annualized vol": ann_vol,
        "Sharpe (rf=0%)": sharpe,
        "Max drawdown": max_dd,
        "Calmar": calmar,
    }


def build_engine() -> BacktestEngine:
    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
        ),
    )
    venue = Venue("XNAS")
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(1_000_000, USD)],
    )
    spy = TestInstrumentProvider.equity("SPY", "XNAS")
    engine.add_instrument(spy)
    return engine


def run_backtest(
    start: str = "2000-01-01",
    end: str | None = None,
    ols_window: int = 18,
    zscore_window: int = 1100,
    buy_threshold: float = 0.7,
    sell_threshold: float = -0.7,
    vol_ma_period: int = 20,
    trade_size: int = 100,
    use_full_capital: bool = False,
) -> None:
    print(f"[1/4] Fetching SPY daily data from FMP (from {start}) ...")
    df = fetch_fmp_daily("SPY", start=start, end=end)
    bar_type = BarType.from_str(BAR_TYPE_STR)
    bars = dataframe_to_bars(df, bar_type)
    print(f"       {len(bars)} bars loaded ({df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()})")

    print("[2/4] Building BacktestEngine ...")
    engine = build_engine()
    engine.add_data(bars)

    config = RSRSStrategyConfig(
        bar_type_str=BAR_TYPE_STR,
        instrument_id_str=INSTRUMENT_ID_STR,
        ols_window=ols_window,
        zscore_window=zscore_window,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        trade_size=trade_size,
        vol_ma_period=vol_ma_period,
        use_full_capital=use_full_capital,
    )
    strategy = RSRSStrategy(config=config)
    engine.add_strategy(strategy)

    print("[3/4] Running backtest ...")
    engine.run()

    print("[4/4] Generating report ...")
    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()

    n_fills = len(fills) if fills is not None else 0
    n_pos = len(positions) if positions is not None else 0
    print(f"       Fills: {n_fills}  |  Positions: {n_pos}")

    _report_and_plot(engine, df, strategy)
    engine.dispose()


def _report_and_plot(
    engine: BacktestEngine,
    df: pd.DataFrame,
    strategy: RSRSStrategy,
) -> None:
    account = engine.trader.generate_account_report(Venue("XNAS"))
    if account is not None and len(account) > 0:
        print("\n=== Account Report ===")
        print(account.to_string())

    fills = engine.trader.generate_order_fills_report()
    if fills is not None and len(fills) > 0:
        fills.to_csv(RESULTS_DIR / "fills.csv")
        print(f"\nFills saved to {RESULTS_DIR / 'fills.csv'}")

    positions = engine.trader.generate_positions_report()
    if positions is not None and len(positions) > 0:
        positions.to_csv(RESULTS_DIR / "positions.csv")
        print(f"Positions saved to {RESULTS_DIR / 'positions.csv'}")

    _plot_equity(df)


def _plot_equity(df: pd.DataFrame) -> None:
    """Plot buy-and-hold equity from the raw price data as a reference."""
    close = df.set_index("date")["close"]
    bh_equity = close / close.iloc[0]

    fig, ax = plt.subplots(figsize=(14, 6))
    bh_equity.plot(ax=ax, label="SPY Buy & Hold", alpha=0.8)
    ax.set_title("RSRS Strategy — SPY Buy & Hold Reference")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curve.png", dpi=150)
    plt.close(fig)
    print(f"Equity curve saved to {RESULTS_DIR / 'equity_curve.png'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RSRS NautilusTrader Backtest")
    p.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--ols-window", type=int, default=18, help="OLS regression window N")
    p.add_argument("--zscore-window", type=int, default=1100, help="Z-score window M")
    p.add_argument("--buy-threshold", type=float, default=0.7)
    p.add_argument("--sell-threshold", type=float, default=-0.7)
    p.add_argument("--vol-ma-period", type=int, default=20, help="Volume MA period")
    p.add_argument("--trade-size", type=int, default=100, help="Shares per trade")
    p.add_argument("--full-capital", action="store_true", help="Use full capital position sizing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_backtest(
        start=args.start,
        end=args.end,
        ols_window=args.ols_window,
        zscore_window=args.zscore_window,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        vol_ma_period=args.vol_ma_period,
        trade_size=args.trade_size,
        use_full_capital=args.full_capital,
    )
