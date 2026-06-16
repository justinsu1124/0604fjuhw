#!/usr/bin/env python3
"""Generate detailed RSRS backtest performance report with charts."""

from __future__ import annotations

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


def run_and_report():
    print("Fetching SPY data from FMP ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    bar_type = BarType.from_str(BAR_TYPE_STR)
    bars = dataframe_to_bars(df, bar_type)
    print(f"  {len(bars)} bars: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    engine = BacktestEngine(
        config=BacktestEngineConfig(trader_id=TraderId("BACKTESTER-001")),
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
    engine.add_data(bars)

    config = RSRSStrategyConfig(
        bar_type_str=BAR_TYPE_STR,
        instrument_id_str=INSTRUMENT_ID_STR,
        ols_window=18,
        zscore_window=1100,
        buy_threshold=0.7,
        sell_threshold=-0.7,
        trade_size=100,
        vol_ma_period=20,
    )
    strategy = RSRSStrategy(config=config)
    engine.add_strategy(strategy)

    print("Running backtest ...")
    engine.run()

    # Extract reports
    fills_df = engine.trader.generate_order_fills_report()
    positions_df = engine.trader.generate_positions_report()
    account_df = engine.trader.generate_account_report(Venue("XNAS"))

    # Build equity curve from account snapshots
    if account_df is not None and len(account_df) > 0:
        equity_series = account_df["total"].astype(float)
    else:
        equity_series = pd.Series(dtype=float)

    # Build buy-and-hold from price data
    df_prices = df.set_index("date").sort_index()
    close = df_prices["close"]

    # Compute strategy metrics
    print("\n" + "=" * 70)
    print("  RSRS × Volume/MA Confirmation — 回測績效報告")
    print("=" * 70)

    print("\n標的: SPY")
    print(f"數據區間: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print(f"交易日數: {len(df)}")
    print("起始資金: $1,000,000")

    if len(equity_series) > 1:
        final_balance = equity_series.iloc[-1]
        print(f"最終資金: ${final_balance:,.0f}")
        print(f"策略總報酬: {(final_balance / 1_000_000 - 1) * 100:.2f}%")

        # Annualized
        n_days = len(equity_series)
        total_ret = final_balance / 1_000_000 - 1
        ann_ret = (1 + total_ret) ** (252 / max(n_days, 1)) - 1

        rets = equity_series.pct_change().dropna()
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

        dd = equity_series / equity_series.cummax() - 1
        max_dd = dd.min()
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")

        print("\n--- 策略績效指標 ---")
        print(f"年化報酬率: {ann_ret * 100:.2f}%")
        print(f"年化波動率: {ann_vol * 100:.2f}%")
        print(f"Sharpe Ratio (rf=0%): {sharpe:.4f}")
        print(f"最大回撤: {max_dd * 100:.2f}%")
        print(f"Calmar Ratio: {calmar:.4f}")

    # Buy-and-hold metrics
    if len(close) > 1:
        bh_total = close.iloc[-1] / close.iloc[0] - 1
        bh_n = len(close)
        bh_ann = (1 + bh_total) ** (252 / max(bh_n, 1)) - 1
        bh_rets = close.pct_change().dropna()
        bh_vol = bh_rets.std() * np.sqrt(252)
        bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else float("nan")
        bh_dd = close / close.cummax() - 1
        bh_maxdd = bh_dd.min()
        bh_calmar = bh_ann / abs(bh_maxdd) if bh_maxdd < 0 else float("nan")

        print("\n--- Buy & Hold 績效指標 ---")
        print(f"總報酬率: {bh_total * 100:.2f}%")
        print(f"年化報酬率: {bh_ann * 100:.2f}%")
        print(f"年化波動率: {bh_vol * 100:.2f}%")
        print(f"Sharpe Ratio: {bh_sharpe:.4f}")
        print(f"最大回撤: {bh_maxdd * 100:.2f}%")
        print(f"Calmar Ratio: {bh_calmar:.4f}")

    # Trades summary
    if positions_df is not None and len(positions_df) > 0:
        print("\n--- 交易統計 ---")
        print(f"總交易筆數: {len(positions_df)}")
        realized_pnl = positions_df["realized_pnl"].astype(str)
        pnl_values = []
        for v in realized_pnl:
            try:
                pnl_values.append(float(v.replace(" USD", "")))
            except (ValueError, AttributeError):
                pass
        if pnl_values:
            pnl_arr = np.array(pnl_values)
            wins = pnl_arr[pnl_arr > 0]
            losses = pnl_arr[pnl_arr <= 0]
            print(f"獲利筆數: {len(wins)}")
            print(f"虧損筆數: {len(losses)}")
            print(f"勝率: {len(wins) / len(pnl_arr) * 100:.1f}%")
            print(f"總已實現損益: ${pnl_arr.sum():,.0f}")
            if len(wins) > 0:
                print(f"平均獲利: ${wins.mean():,.0f}")
            if len(losses) > 0:
                print(f"平均虧損: ${losses.mean():,.0f}")
            if len(losses) > 0 and losses.mean() != 0:
                print(f"盈虧比: {abs(wins.mean() / losses.mean()):.2f}")

    if fills_df is not None and len(fills_df) > 0:
        print(f"總成交筆數(fills): {len(fills_df)}")

    # --- Charts ---
    print("\nGenerating charts ...")

    # 1. Equity curve comparison
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # Normalize buy-and-hold to $1M start
    bh_equity = close / close.iloc[0] * 1_000_000

    if len(equity_series) > 1:
        equity_series.plot(ax=axes[0], label="RSRS Strategy", linewidth=1.5)
    bh_equity.plot(ax=axes[0], label="SPY Buy & Hold", alpha=0.7, linewidth=1.5)
    axes[0].set_title("累積權益曲線 — RSRS Strategy vs Buy & Hold", fontsize=14)
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend(fontsize=12)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # 2. Drawdown
    if len(equity_series) > 1:
        strat_dd = (equity_series / equity_series.cummax() - 1) * 100
        strat_dd.plot(ax=axes[1], label="RSRS Strategy DD", color="red", alpha=0.7)
    bh_dd_pct = (close / close.cummax() - 1) * 100
    bh_dd_pct.plot(ax=axes[1], label="Buy & Hold DD", color="blue", alpha=0.5)
    axes[1].set_title("回撤 (Drawdown %)", fontsize=14)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend(fontsize=12)
    axes[1].fill_between(bh_dd_pct.index, bh_dd_pct.values, alpha=0.1, color="blue")

    # 3. Trade P&L
    if pnl_values:
        colors = ["green" if v > 0 else "red" for v in pnl_values]
        axes[2].bar(range(len(pnl_values)), pnl_values, color=colors, alpha=0.8)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_title("各筆交易損益 (Realized P&L)", fontsize=14)
        axes[2].set_xlabel("Trade #")
        axes[2].set_ylabel("P&L ($)")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "performance_report.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nCharts saved to {chart_path}")

    # --- RSRS Signal chart ---
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    # Rebuild RSRS signal from indicator data
    from rsrs_indicator import RSRSIndicator
    rsrs = RSRSIndicator(ols_window=18, zscore_window=1100)
    signals = []
    dates = []
    for _, row in df_prices.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        if rsrs.initialized:
            signals.append(rsrs.signal_value)
            dates.append(row.name)

    if signals:
        sig_series = pd.Series(signals, index=dates)
        sig_series.plot(ax=ax2, color="tab:orange", alpha=0.8, label="RSRS Signal")
        ax2.axhline(0.7, color="green", ls="--", lw=1, label="Buy threshold (0.7)")
        ax2.axhline(-0.7, color="red", ls="--", lw=1, label="Sell threshold (-0.7)")
        ax2.set_title("RSRS Right-Skewed Signal", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.set_ylabel("Signal Value")

    plt.tight_layout()
    signal_path = RESULTS_DIR / "rsrs_signal.png"
    fig2.savefig(signal_path, dpi=150)
    plt.close(fig2)
    print(f"Signal chart saved to {signal_path}")

    engine.dispose()
    print("\nDone!")


if __name__ == "__main__":
    run_and_report()
