#!/usr/bin/env python3
"""Compare multiple Volume/MA confirmation filters on RSRS strategy.

Usage
-----
    FMP_API_KEY=<key> python compare_filters.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import fetch_fmp_daily
from rsrs_indicator import RSRSIndicator

sns.set_theme(style="whitegrid")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLS_WINDOW = 18
ZSCORE_WINDOW = 1100
BUY_THRESHOLD = 0.7
SELL_THRESHOLD = -0.7
TC_BPS = 5
TC_RATE = TC_BPS / 10_000


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run RSRS indicator and attach signals + price/volume helpers to df."""
    df_p = df.set_index("date").sort_index().copy()

    rsrs = RSRSIndicator(ols_window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    signals = []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        signals.append(rsrs.signal_value if rsrs.initialized else float("nan"))

    df_p["signal"] = signals
    df_p["daily_ret"] = df_p["open"].pct_change()

    # Pre-compute volume SMAs
    for w in [10, 20, 50]:
        df_p[f"vol_sma_{w}"] = df_p["volume"].rolling(w).mean()

    # Pre-compute price SMAs
    for w in [10, 20, 50, 200]:
        df_p[f"price_sma_{w}"] = df_p["close"].rolling(w).mean()

    return df_p


def generate_positions(signal: pd.Series, buy_gate: pd.Series, buy: float, sell: float) -> pd.Series:
    """Stateful position: 1=long, 0=cash. buy_gate must be True to enter."""
    pos = []
    prev = 0
    for s, gate in zip(signal, buy_gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > buy and gate:
            prev = 1
            pos.append(prev)
        elif s < sell:
            prev = 0
            pos.append(prev)
        else:
            pos.append(prev)
    return pd.Series(pos, index=signal.index, dtype=float)


def backtest_variant(df_p: pd.DataFrame, buy_gate: pd.Series, label: str) -> dict:
    """Run a single backtest variant and return stats + equity series."""
    pos_raw = generate_positions(df_p["signal"], buy_gate, BUY_THRESHOLD, SELL_THRESHOLD)
    position = pos_raw.shift(1).fillna(0)

    delta = position.diff().abs()
    delta.iloc[0] = abs(position.iloc[0])
    tc = delta.fillna(0) * TC_RATE

    strat_ret = position * df_p["daily_ret"] - tc

    # Trim to first valid signal
    first_valid = df_p["signal"].first_valid_index()
    if first_valid is not None:
        strat_ret = strat_ret.loc[first_valid:]
        position = position.loc[first_valid:]
        mkt_ret = df_p["daily_ret"].loc[first_valid:]
    else:
        mkt_ret = df_p["daily_ret"]

    eq_strat = (1 + strat_ret.fillna(0)).cumprod()
    eq_mkt = (1 + mkt_ret.fillna(0)).cumprod()

    # Stats
    rets = eq_strat.pct_change().dropna()
    n = len(rets)
    if n < 2:
        return {"label": label, "equity": eq_strat}
    total = eq_strat.iloc[-1] / eq_strat.iloc[0] - 1
    ann = (1 + total) ** (252 / n) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else float("nan")
    dd = eq_strat / eq_strat.cummax() - 1
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")

    # Count trades
    trades = (pos_raw.diff().abs() > 0).sum()

    return {
        "label": label,
        "equity": eq_strat,
        "equity_mkt": eq_mkt,
        "Total return": total,
        "Ann return": ann,
        "Ann vol": vol,
        "Sharpe": sharpe,
        "Max DD": maxdd,
        "Calmar": calmar,
        "Trades": int(trades),
    }


def main() -> None:
    print("Fetching SPY data from FMP ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    print("Computing RSRS signals and indicators ...")
    df_p = compute_signals(df)

    # Define filter variants
    always_true = pd.Series(True, index=df_p.index)

    variants: list[tuple[str, pd.Series]] = [
        ("No filter (baseline)", always_true),
        ("Vol > SMA(vol,10)", df_p["volume"] > df_p["vol_sma_10"]),
        ("Vol > SMA(vol,20)", df_p["volume"] > df_p["vol_sma_20"]),
        ("Vol > SMA(vol,50)", df_p["volume"] > df_p["vol_sma_50"]),
        ("Price > SMA(close,20)", df_p["close"] > df_p["price_sma_20"]),
        ("Price > SMA(close,50)", df_p["close"] > df_p["price_sma_50"]),
        ("Price > SMA(close,200)", df_p["close"] > df_p["price_sma_200"]),
        ("Vol>SMA20 & Price>SMA20", (df_p["volume"] > df_p["vol_sma_20"]) & (df_p["close"] > df_p["price_sma_20"])),
        ("Vol>SMA20 & Price>SMA50", (df_p["volume"] > df_p["vol_sma_20"]) & (df_p["close"] > df_p["price_sma_50"])),
    ]

    print(f"\nRunning {len(variants)} filter variants ...")
    results = []
    for label, gate in variants:
        gate = gate.fillna(False)
        res = backtest_variant(df_p, gate, label)
        results.append(res)
        if "Sharpe" in res:
            print(f"  {label:35s}  Sharpe={res['Sharpe']:.3f}  Total={res['Total return']:.2%}  MaxDD={res['Max DD']:.2%}  Trades={res['Trades']}")

    # --- Summary table ---
    table_rows = []
    for r in results:
        if "Sharpe" in r:
            table_rows.append({
                "Filter": r["label"],
                "Total Return": f"{r['Total return']:.1%}",
                "Ann Return": f"{r['Ann return']:.2%}",
                "Ann Vol": f"{r['Ann vol']:.2%}",
                "Sharpe": f"{r['Sharpe']:.3f}",
                "Max DD": f"{r['Max DD']:.2%}",
                "Calmar": f"{r['Calmar']:.3f}",
                "Trades": r["Trades"],
            })
    summary = pd.DataFrame(table_rows)
    print("\n" + "=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)
    summary.to_csv(RESULTS_DIR / "filter_comparison.csv", index=False)

    # --- Charts ---
    print("\nGenerating charts ...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Equity curves
    for r in results:
        if "equity" in r:
            r["equity"].plot(ax=axes[0], label=r["label"], alpha=0.8, linewidth=1.2)
    if "equity_mkt" in results[0]:
        results[0]["equity_mkt"].plot(ax=axes[0], label="SPY Buy & Hold", color="black", linewidth=1.5, ls="--")
    axes[0].set_title("RSRS Strategy — Filter Comparison (Cumulative Equity)", fontsize=14)
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(fontsize=8, loc="upper left")

    # 2. Bar chart of Sharpe ratios
    labels = [r["label"] for r in results if "Sharpe" in r]
    sharpes = [r["Sharpe"] for r in results if "Sharpe" in r]
    colors = ["tab:green" if s == max(sharpes) else "tab:blue" for s in sharpes]
    axes[1].barh(labels, sharpes, color=colors, alpha=0.8)
    axes[1].set_xlabel("Sharpe Ratio")
    axes[1].set_title("Sharpe Ratio by Filter Variant", fontsize=14)
    for i, v in enumerate(sharpes):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    chart_path = RESULTS_DIR / "filter_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to {chart_path}")

    # --- Drawdown comparison ---
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    for r in results:
        if "equity" in r:
            eq = r["equity"]
            dd = (eq / eq.cummax() - 1) * 100
            dd.plot(ax=ax2, label=r["label"], alpha=0.6, linewidth=1)
    ax2.set_title("Drawdown Comparison (%)", fontsize=14)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    dd_path = RESULTS_DIR / "filter_drawdown.png"
    fig2.savefig(dd_path, dpi=150)
    plt.close(fig2)
    print(f"Drawdown chart saved to {dd_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
