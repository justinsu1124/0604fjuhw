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
    use_volume_filter: bool = True,
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
        use_volume_filter=use_volume_filter,
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

    _report_and_plot(engine, df, strategy, ols_window, zscore_window, buy_threshold, sell_threshold)
    engine.dispose()


def _report_and_plot(
    engine: BacktestEngine,
    df: pd.DataFrame,
    strategy: RSRSStrategy,
    ols_window: int = 18,
    zscore_window: int = 1100,
    buy_threshold: float = 0.7,
    sell_threshold: float = -0.7,
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

    _plot_equity(df, ols_window, zscore_window, buy_threshold, sell_threshold)


def _compute_daily_equity(
    df: pd.DataFrame,
    ols_window: int,
    zscore_window: int,
    buy_threshold: float,
    sell_threshold: float,
) -> pd.DataFrame:
    """Compute daily mark-to-market equity using the same logic as the reference notebook.

    Returns a DataFrame with columns: signal, position, equity_market, equity_strategy.
    Uses open-to-open returns with 1-day implementation lag and 5 bps transaction cost.
    """
    from rsrs_indicator import RSRSIndicator

    df_p = df.set_index("date").sort_index()

    rsrs = RSRSIndicator(ols_window=ols_window, zscore_window=zscore_window)
    signals = []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        signals.append(rsrs.signal_value if rsrs.initialized else float("nan"))

    bt = pd.DataFrame({"signal": signals}, index=df_p.index)

    pos = []
    prev = 0
    for s in bt["signal"]:
        if pd.isna(s):
            pos.append(prev)
        elif s > buy_threshold:
            prev = 1
            pos.append(prev)
        elif s < sell_threshold:
            prev = 0
            pos.append(prev)
        else:
            pos.append(prev)
    bt["position"] = pos
    bt["position"] = bt["position"].shift(1).fillna(0)  # 1-day lag

    daily_ret = df_p["open"].pct_change()
    bt["market_ret"] = daily_ret

    tc_rate = 5 / 10_000  # 5 bps
    delta = bt["position"].diff().abs()
    delta.iloc[0] = abs(bt["position"].iloc[0])
    bt["tc"] = delta.fillna(0) * tc_rate

    bt["strategy_ret"] = bt["position"] * bt["market_ret"] - bt["tc"]

    # Trim to start from first valid signal (after warmup)
    first_valid = bt["signal"].first_valid_index()
    if first_valid is not None:
        bt = bt.loc[first_valid:]

    bt["equity_market"] = (1 + bt["market_ret"].fillna(0)).cumprod()
    bt["equity_strategy"] = (1 + bt["strategy_ret"].fillna(0)).cumprod()

    return bt.dropna(subset=["market_ret"])


def _plot_equity(
    df: pd.DataFrame,
    ols_window: int = 18,
    zscore_window: int = 1100,
    buy_threshold: float = 0.7,
    sell_threshold: float = -0.7,
) -> None:
    """Generate 2-panel chart matching the reference notebook format."""
    bt = _compute_daily_equity(df, ols_window, zscore_window, buy_threshold, sell_threshold)

    # --- Performance stats ---
    eq_strat = bt["equity_strategy"]
    eq_mkt = bt["equity_market"]

    def _stats(eq: pd.Series) -> dict:
        rets = eq.pct_change().dropna()
        n = len(rets)
        if n < 2:
            return {}
        total = eq.iloc[-1] / eq.iloc[0] - 1
        ann = (1 + total) ** (252 / n) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else float("nan")
        dd = eq / eq.cummax() - 1
        maxdd = dd.min()
        calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
        return {"Total": total, "Ann ret": ann, "Ann vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar}

    s_strat = _stats(eq_strat)
    s_mkt = _stats(eq_mkt)
    print("\n=== Daily Mark-to-Market Performance ===")
    print(f"{'':20s} {'RSRS':>12s} {'Buy&Hold':>12s}")
    for k in s_strat:
        print(f"  {k:18s} {s_strat[k]:12.4f} {s_mkt.get(k, 0):12.4f}")

    # --- 2-panel chart (matching notebook) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    bt[["equity_market", "equity_strategy"]].plot(
        ax=axes[0], title="Cumulative equity: SPY vs RSRS timing",
    )
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(["SPY buy & hold", "RSRS right-skew"])

    bt["signal"].plot(ax=axes[1], color="tab:orange", alpha=0.8, label="Signal")
    axes[1].axhline(buy_threshold, color="green", ls="--", lw=1, label="Buy threshold")
    axes[1].axhline(sell_threshold, color="red", ls="--", lw=1, label="Sell threshold")
    bt["position"].plot(ax=axes[1], color="tab:blue", alpha=0.4, label="Position")
    axes[1].set_title("Right-skewed RSRS signal and position")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "equity_curve.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nEquity curve saved to {chart_path}")


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
    p.add_argument("--no-volume-filter", action="store_true", help="Disable volume/MA confirmation filter")
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
        use_volume_filter=not args.no_volume_filter,
    )
