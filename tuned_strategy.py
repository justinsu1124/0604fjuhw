#!/usr/bin/env python3
"""Tuned strategy: β_z×R²×β + R²>0.9 with asymmetric threshold search.

Focuses on buy_threshold >= sell_threshold (buy higher, sell lower),
e.g. buy=0.7 sell=-0.5 as suggested.

Usage
-----
    FMP_API_KEY=<key> python tuned_strategy.py
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


def compute_data(df: pd.DataFrame) -> pd.DataFrame:
    df_p = df.set_index("date").sort_index().copy()
    rsrs = RSRSIndicator(ols_window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    sigs, rsqs = [], []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        sigs.append(rsrs.signal_value if rsrs.initialized else float("nan"))
        rsqs.append(rsrs.rsquare)
    df_p["signal"] = sigs
    df_p["rsquare"] = rsqs
    df_p["daily_ret"] = df_p["open"].pct_change()
    return df_p


def backtest(df_p: pd.DataFrame, buy_t: float, sell_t: float) -> dict:
    signal = df_p["signal"]
    gate = df_p["rsquare"] > 0.9

    pos = []
    prev = 0
    for s, g in zip(signal, gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > buy_t and g:
            prev = 1
            pos.append(prev)
        elif s < sell_t:
            prev = 0
            pos.append(prev)
        else:
            pos.append(prev)

    pos_series = pd.Series(pos, index=signal.index, dtype=float)
    position = pos_series.shift(1).fillna(0)
    delta = position.diff().abs()
    delta.iloc[0] = abs(position.iloc[0])
    tc = delta.fillna(0) * TC_RATE
    strat_ret = position * df_p["daily_ret"] - tc

    first_valid = signal.first_valid_index()
    if first_valid is not None:
        strat_ret = strat_ret.loc[first_valid:]
        mkt_ret = df_p["daily_ret"].loc[first_valid:]
    else:
        mkt_ret = df_p["daily_ret"]

    eq = (1 + strat_ret.fillna(0)).cumprod()
    eq_mkt = (1 + mkt_ret.fillna(0)).cumprod()

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
    trades = int((pos_series.diff().abs() > 0).sum())

    return {
        "Total": total, "Ann ret": ann, "Ann vol": vol,
        "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar,
        "Trades": trades, "equity": eq, "equity_mkt": eq_mkt,
    }


def main() -> None:
    print("Fetching SPY data ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars")

    print("Computing β_z×R²×β signal + R² ...")
    df_p = compute_data(df)

    # --- Asymmetric threshold grid: buy higher, sell lower ---
    buy_range = np.arange(0.5, 1.15, 0.1)   # 0.5 to 1.1
    sell_range = np.arange(-0.3, -1.05, -0.1)  # -0.3 to -1.0

    print(f"Running {len(buy_range)} × {len(sell_range)} = {len(buy_range) * len(sell_range)} threshold combos ...")

    all_results = []
    sharpe_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    calmar_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    total_grid = np.full((len(sell_range), len(buy_range)), np.nan)
    maxdd_grid = np.full((len(sell_range), len(buy_range)), np.nan)

    for i, sell_t in enumerate(sell_range):
        for j, buy_t in enumerate(buy_range):
            res = backtest(df_p, buy_t, sell_t)
            if res:
                sharpe_grid[i, j] = res["Sharpe"]
                calmar_grid[i, j] = res["Calmar"]
                total_grid[i, j] = res["Total"]
                maxdd_grid[i, j] = res["MaxDD"]
                all_results.append({
                    "Buy": round(buy_t, 1), "Sell": round(sell_t, 1),
                    **{k: v for k, v in res.items() if k not in ("equity", "equity_mkt")},
                })

    rdf = pd.DataFrame(all_results).sort_values("Sharpe", ascending=False)

    # Print top 15
    print("\n" + "=" * 95)
    print("β_z×R²×β + R²>0.9 — Asymmetric Threshold Search (buy higher, sell lower)")
    print("=" * 95)
    top = rdf.head(15).copy()
    fmt = {"Total": "{:.1%}", "Ann ret": "{:.2%}", "Ann vol": "{:.2%}",
           "Sharpe": "{:.3f}", "MaxDD": "{:.2%}", "Calmar": "{:.3f}"}
    for c, f in fmt.items():
        top[c] = top[c].map(f.format)
    print(top.to_string(index=False))

    # Highlight the user's requested combo
    user_combo = rdf[(rdf["Buy"] == 0.7) & (rdf["Sell"] == -0.5)]
    if not user_combo.empty:
        r = user_combo.iloc[0]
        print("\n>>> User requested Buy=0.7 Sell=-0.5:")
        print(f"    Total={r['Total']:.1%}  Sharpe={r['Sharpe']:.3f}  MaxDD={r['MaxDD']:.2%}  Calmar={r['Calmar']:.3f}  Trades={r['Trades']}")

    ref_combo = rdf[(rdf["Buy"] == 0.7) & (rdf["Sell"] == -0.8)]
    if not ref_combo.empty:
        r = ref_combo.iloc[0]
        print("\n>>> Reference Buy=0.7 Sell=-0.8:")
        print(f"    Total={r['Total']:.1%}  Sharpe={r['Sharpe']:.3f}  MaxDD={r['MaxDD']:.2%}  Calmar={r['Calmar']:.3f}  Trades={r['Trades']}")

    rdf_save = rdf.copy()
    for c, f in fmt.items():
        rdf_save[c] = rdf_save[c].map(f.format)
    rdf_save.to_csv(RESULTS_DIR / "tuned_thresholds.csv", index=False)

    # --- Heatmaps ---
    print("\nGenerating charts ...")
    buy_labels = [f"{v:.1f}" for v in buy_range]
    sell_labels = [f"{v:.1f}" for v in sell_range]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    sns.heatmap(sharpe_grid, ax=axes[0, 0], annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[0, 0].set_title("Sharpe Ratio", fontsize=13)
    axes[0, 0].set_xlabel("Buy Threshold")
    axes[0, 0].set_ylabel("Sell Threshold")

    sns.heatmap(total_grid * 100, ax=axes[0, 1], annot=True, fmt=".0f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[0, 1].set_title("Total Return (%)", fontsize=13)
    axes[0, 1].set_xlabel("Buy Threshold")
    axes[0, 1].set_ylabel("Sell Threshold")

    sns.heatmap(maxdd_grid * 100, ax=axes[1, 0], annot=True, fmt=".1f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[1, 0].set_title("Max Drawdown (%)", fontsize=13)
    axes[1, 0].set_xlabel("Buy Threshold")
    axes[1, 0].set_ylabel("Sell Threshold")

    sns.heatmap(calmar_grid, ax=axes[1, 1], annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=buy_labels, yticklabels=sell_labels)
    axes[1, 1].set_title("Calmar Ratio", fontsize=13)
    axes[1, 1].set_xlabel("Buy Threshold")
    axes[1, 1].set_ylabel("Sell Threshold")

    plt.suptitle("β_z×R²×β + R²>0.9 — Asymmetric Threshold Grid", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "tuned_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Key equity curves ---
    key_combos = [
        (0.7, -0.5, "Buy=0.7 Sell=-0.5 (requested)"),
        (0.7, -0.8, "Buy=0.7 Sell=-0.8 (reference)"),
        (0.7, -0.7, "Buy=0.7 Sell=-0.7 (symmetric)"),
    ]
    # Add top 3 from grid
    for _, row in rdf.head(3).iterrows():
        key = (row["Buy"], row["Sell"])
        if key not in [(k[0], k[1]) for k in key_combos]:
            key_combos.append((row["Buy"], row["Sell"], f"Buy={row['Buy']:.1f} Sell={row['Sell']:.1f} (top)"))

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    plotted_mkt = False
    for buy_t, sell_t, label in key_combos:
        res = backtest(df_p, buy_t, sell_t)
        if res:
            lbl = f"{label} S={res['Sharpe']:.3f}"
            res["equity"].plot(ax=ax2, label=lbl, linewidth=1.3, alpha=0.85)
            if not plotted_mkt:
                res["equity_mkt"].plot(ax=ax2, label="SPY Buy & Hold", color="black", ls="--", lw=1.5)
                plotted_mkt = True

    ax2.set_title("β_z×R²×β + R²>0.9 — Key Threshold Comparisons", fontsize=14)
    ax2.set_ylabel("Growth of $1")
    ax2.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "tuned_equity.png", dpi=150)
    plt.close(fig2)

    print(f"Charts saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
