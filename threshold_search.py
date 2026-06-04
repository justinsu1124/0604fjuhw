#!/usr/bin/env python3
"""Asymmetric buy/sell threshold grid search for RSRS strategy.

Scans combinations of buy_threshold and sell_threshold independently,
then generates heatmaps and a ranked table of top combinations.

Usage
-----
    FMP_API_KEY=<key> python threshold_search.py
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
TC_RATE = 5 / 10_000


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Run RSRS indicator on price data."""
    df_p = df.set_index("date").sort_index().copy()
    rsrs = RSRSIndicator(ols_window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    signals = []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        signals.append(rsrs.signal_value if rsrs.initialized else float("nan"))
    df_p["signal"] = signals
    df_p["daily_ret"] = df_p["open"].pct_change()
    return df_p


def backtest_threshold(
    df_p: pd.DataFrame,
    buy_thresh: float,
    sell_thresh: float,
) -> dict:
    """Run backtest with given buy/sell thresholds. Returns stats dict."""
    signal = df_p["signal"]

    pos = []
    prev = 0
    for s in signal:
        if pd.isna(s):
            pos.append(prev)
        elif s > buy_thresh:
            prev = 1
            pos.append(prev)
        elif s < sell_thresh:
            prev = 0
            pos.append(prev)
        else:
            pos.append(prev)
    position = pd.Series(pos, index=signal.index, dtype=float).shift(1).fillna(0)

    delta = position.diff().abs()
    delta.iloc[0] = abs(position.iloc[0])
    tc = delta.fillna(0) * TC_RATE

    strat_ret = position * df_p["daily_ret"] - tc

    first_valid = signal.first_valid_index()
    if first_valid is not None:
        strat_ret = strat_ret.loc[first_valid:]

    eq = (1 + strat_ret.fillna(0)).cumprod()

    rets = eq.pct_change().dropna()
    n = len(rets)
    if n < 2:
        return {"Sharpe": float("nan"), "Total": float("nan"), "MaxDD": float("nan"),
                "Ann ret": float("nan"), "Calmar": float("nan"), "Trades": 0, "equity": eq}

    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (252 / n) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else float("nan")
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    trades = int((pd.Series(pos).diff().abs() > 0).sum())

    return {
        "Sharpe": sharpe, "Total": total, "MaxDD": maxdd,
        "Ann ret": ann, "Ann vol": vol, "Calmar": calmar,
        "Trades": trades, "equity": eq,
    }


def main() -> None:
    print("Fetching SPY data from FMP ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    print("Computing RSRS signals ...")
    df_p = compute_signals(df)

    # Grid: buy thresholds 0.3 to 1.2, sell thresholds -0.3 to -1.2
    buy_range = np.arange(0.3, 1.25, 0.1)
    sell_range = np.arange(-0.3, -1.25, -0.1)

    print(f"Running grid search: {len(buy_range)} × {len(sell_range)} = {len(buy_range) * len(sell_range)} combinations ...")

    sharpe_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    total_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    maxdd_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    calmar_grid = np.full((len(sell_range), len(buy_range)), np.nan)

    all_results = []

    for i, sell_t in enumerate(sell_range):
        for j, buy_t in enumerate(buy_range):
            res = backtest_threshold(df_p, buy_t, sell_t)
            sharpe_grid[i, j] = res["Sharpe"]
            total_grid[i, j] = res["Total"]
            maxdd_grid[i, j] = res["MaxDD"]
            calmar_grid[i, j] = res["Calmar"]
            all_results.append({
                "Buy": round(buy_t, 1),
                "Sell": round(sell_t, 1),
                **{k: v for k, v in res.items() if k != "equity"},
            })

    results_df = pd.DataFrame(all_results).sort_values("Sharpe", ascending=False)

    # Print top 15
    print("\n" + "=" * 90)
    print("Top 15 threshold combinations by Sharpe Ratio:")
    print("=" * 90)
    top15 = results_df.head(15).copy()
    top15["Total"] = top15["Total"].map(lambda x: f"{x:.1%}")
    top15["Ann ret"] = top15["Ann ret"].map(lambda x: f"{x:.2%}")
    top15["Ann vol"] = top15["Ann vol"].map(lambda x: f"{x:.2%}")
    top15["Sharpe"] = top15["Sharpe"].map(lambda x: f"{x:.3f}")
    top15["MaxDD"] = top15["MaxDD"].map(lambda x: f"{x:.2%}")
    top15["Calmar"] = top15["Calmar"].map(lambda x: f"{x:.3f}")
    print(top15.to_string(index=False))

    results_df.to_csv(RESULTS_DIR / "threshold_grid.csv", index=False)

    # --- Heatmaps ---
    print("\nGenerating heatmaps ...")
    buy_labels = [f"{v:.1f}" for v in buy_range]
    sell_labels = [f"{v:.1f}" for v in sell_range]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # Sharpe heatmap
    sns.heatmap(sharpe_grid, ax=axes[0, 0], annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[0, 0].set_title("Sharpe Ratio", fontsize=13)
    axes[0, 0].set_xlabel("Buy Threshold")
    axes[0, 0].set_ylabel("Sell Threshold")

    # Total return heatmap
    sns.heatmap(total_grid * 100, ax=axes[0, 1], annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[0, 1].set_title("Total Return (%)", fontsize=13)
    axes[0, 1].set_xlabel("Buy Threshold")
    axes[0, 1].set_ylabel("Sell Threshold")

    # Max DD heatmap
    sns.heatmap(maxdd_grid * 100, ax=axes[1, 0], annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[1, 0].set_title("Max Drawdown (%)", fontsize=13)
    axes[1, 0].set_xlabel("Buy Threshold")
    axes[1, 0].set_ylabel("Sell Threshold")

    # Calmar heatmap
    sns.heatmap(calmar_grid, ax=axes[1, 1], annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[1, 1].set_title("Calmar Ratio", fontsize=13)
    axes[1, 1].set_xlabel("Buy Threshold")
    axes[1, 1].set_ylabel("Sell Threshold")

    plt.suptitle("RSRS Asymmetric Threshold Grid Search (SPY 2004-2026)", fontsize=15, y=1.01)
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "threshold_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {heatmap_path}")

    # --- Top 5 equity curves ---
    top5_params = results_df.head(5)[["Buy", "Sell"]].values.tolist()
    fig2, ax2 = plt.subplots(figsize=(14, 6))

    for buy_t, sell_t in top5_params:
        res = backtest_threshold(df_p, buy_t, sell_t)
        label = f"Buy={buy_t:.1f} Sell={sell_t:.1f} (S={res['Sharpe']:.3f})"
        res["equity"].plot(ax=ax2, label=label, linewidth=1.3, alpha=0.85)

    # Baseline (symmetric 0.7/-0.7)
    base = backtest_threshold(df_p, 0.7, -0.7)
    base["equity"].plot(ax=ax2, label=f"Baseline 0.7/-0.7 (S={base['Sharpe']:.3f})", color="black", ls="--", linewidth=1.5)

    # Buy & hold
    first_valid = df_p["signal"].first_valid_index()
    mkt_ret = df_p["daily_ret"].loc[first_valid:]
    eq_mkt = (1 + mkt_ret.fillna(0)).cumprod()
    eq_mkt.plot(ax=ax2, label="SPY Buy & Hold", color="gray", ls=":", linewidth=1.5)

    ax2.set_title("Top 5 Asymmetric Thresholds vs Baseline", fontsize=14)
    ax2.set_ylabel("Growth of $1")
    ax2.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    top5_path = RESULTS_DIR / "threshold_top5.png"
    fig2.savefig(top5_path, dpi=150)
    plt.close(fig2)
    print(f"Top 5 chart saved to {top5_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
