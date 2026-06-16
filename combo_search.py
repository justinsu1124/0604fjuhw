#!/usr/bin/env python3
"""Combination search: best filters × best signals × best thresholds.

Crosses:
  - 5 signal variants
  - 6 filter variants (incl. no-filter)
  - 5 threshold pairs
= 150 combinations, ranked by Sharpe, Calmar, Total return.

Usage
-----
    FMP_API_KEY=<key> python combo_search.py
"""

from __future__ import annotations

from itertools import product
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


# ---------------------------------------------------------------------------
# Compute all indicators once
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    df_p = df.set_index("date").sort_index().copy()

    rsrs = RSRSIndicator(ols_window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    rows = []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        rows.append({
            "sig_standard": rsrs.signal_standard if rsrs.initialized else float("nan"),
            "sig_r2adj": rsrs.signal_r2adj if rsrs.initialized else float("nan"),
            "sig_right_skew": rsrs.signal_value if rsrs.initialized else float("nan"),
            "sig_tval_adj": rsrs.signal_tvalue_adj if rsrs.initialized else float("nan"),
            "sig_tval_right": rsrs.signal_tvalue_right if rsrs.initialized else float("nan"),
            "rsquare": rsrs.rsquare,
        })
    extra = pd.DataFrame(rows, index=df_p.index)
    df_p = pd.concat([df_p, extra], axis=1)
    df_p["daily_ret"] = df_p["open"].pct_change()

    # ATR(20)
    tr = pd.concat([
        df_p["high"] - df_p["low"],
        (df_p["high"] - df_p["close"].shift(1)).abs(),
        (df_p["low"] - df_p["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_p["atr_20"] = tr.rolling(20).mean()
    df_p["atr_20_median"] = df_p["atr_20"].rolling(252).median()

    # SMA
    df_p["sma_50"] = df_p["close"].rolling(50).mean()
    df_p["sma_200"] = df_p["close"].rolling(200).mean()
    df_p["sma50_slope"] = df_p["sma_50"] - df_p["sma_50"].shift(10)

    # Volume SMA
    df_p["vol_sma_50"] = df_p["volume"].rolling(50).mean()

    return df_p


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def backtest(signal: pd.Series, buy_gate: pd.Series, daily_ret: pd.Series,
             buy_thresh: float, sell_thresh: float) -> dict:
    pos = []
    prev = 0
    for s, gate in zip(signal, buy_gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > buy_thresh and gate:
            prev = 1
            pos.append(prev)
        elif s < sell_thresh:
            prev = 0
            pos.append(prev)
        else:
            pos.append(prev)
    pos_series = pd.Series(pos, index=signal.index, dtype=float)
    position = pos_series.shift(1).fillna(0)

    delta = position.diff().abs()
    delta.iloc[0] = abs(position.iloc[0])
    tc = delta.fillna(0) * TC_RATE
    strat_ret = position * daily_ret - tc

    first_valid = signal.first_valid_index()
    if first_valid is not None:
        strat_ret = strat_ret.loc[first_valid:]
        mkt_ret = daily_ret.loc[first_valid:]
    else:
        mkt_ret = daily_ret

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching SPY data ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars")

    print("Computing indicators ...")
    df_p = compute_all(df)

    always_true = pd.Series(True, index=df_p.index)

    # --- Dimensions ---
    signals: list[tuple[str, pd.Series]] = [
        ("β_z", df_p["sig_standard"]),
        ("β_z×R²", df_p["sig_r2adj"]),
        ("β_z×R²×β", df_p["sig_right_skew"]),
        ("t×β_z/μt", df_p["sig_tval_adj"]),
        ("t×β_z×β/μt", df_p["sig_tval_right"]),
    ]

    filters: list[tuple[str, pd.Series]] = [
        ("None", always_true),
        ("R²>0.9", df_p["rsquare"] > 0.9),
        ("Close>SMA200", df_p["close"] > df_p["sma_200"]),
        ("SMA200+slope", (df_p["close"] > df_p["sma_200"]) & (df_p["sma50_slope"] > 0)),
        ("ATR<1.5×", df_p["atr_20"] < 1.5 * df_p["atr_20_median"]),
        ("Vol>SMA50", df_p["volume"] > df_p["vol_sma_50"]),
    ]

    thresholds: list[tuple[str, float, float]] = [
        ("0.7/-0.7", 0.7, -0.7),
        ("0.4/-0.4", 0.4, -0.4),
        ("0.7/-0.8", 0.7, -0.8),
        ("0.5/-0.8", 0.5, -0.8),
        ("0.4/-0.8", 0.4, -0.8),
    ]

    total_combos = len(signals) * len(filters) * len(thresholds)
    print(f"\nRunning {len(signals)} signals × {len(filters)} filters × {len(thresholds)} thresholds = {total_combos} combos ...")

    all_results = []
    done = 0
    for (sig_name, sig_series), (filt_name, filt_gate), (thresh_name, buy_t, sell_t) in product(signals, filters, thresholds):
        gate = filt_gate.fillna(False)
        res = backtest(sig_series, gate, df_p["daily_ret"], buy_t, sell_t)
        if res:
            res["Signal"] = sig_name
            res["Filter"] = filt_name
            res["Threshold"] = thresh_name
            all_results.append(res)
        done += 1
        if done % 30 == 0:
            print(f"  {done}/{total_combos} ...")

    print(f"  {done}/{total_combos} done.\n")

    # --- Build results DataFrame ---
    rows = []
    for r in all_results:
        rows.append({
            "Signal": r["Signal"],
            "Filter": r["Filter"],
            "Threshold": r["Threshold"],
            "Total": r["Total"],
            "Ann ret": r["Ann ret"],
            "Ann vol": r["Ann vol"],
            "Sharpe": r["Sharpe"],
            "MaxDD": r["MaxDD"],
            "Calmar": r["Calmar"],
            "Trades": r["Trades"],
        })
    rdf = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)

    # --- Top 20 by Sharpe ---
    print("=" * 110)
    print("Top 20 Combinations by Sharpe Ratio:")
    print("=" * 110)
    top20 = rdf.head(20).copy()
    fmt_cols = {"Total": "{:.1%}", "Ann ret": "{:.2%}", "Ann vol": "{:.2%}",
                "Sharpe": "{:.3f}", "MaxDD": "{:.2%}", "Calmar": "{:.3f}"}
    for c, fmt in fmt_cols.items():
        top20[c] = top20[c].map(fmt.format)
    print(top20.to_string(index=False))

    rdf_save = rdf.copy()
    for c, fmt in fmt_cols.items():
        rdf_save[c] = rdf_save[c].map(fmt.format)
    rdf_save.to_csv(RESULTS_DIR / "combo_all.csv", index=False)

    # --- Top 10 by Calmar ---
    print("\n" + "=" * 110)
    print("Top 10 Combinations by Calmar Ratio:")
    print("=" * 110)
    rdf_cal = rdf.sort_values("Calmar", ascending=False).head(10).copy()
    for c, fmt in fmt_cols.items():
        rdf_cal[c] = rdf_cal[c].map(fmt.format)
    print(rdf_cal.to_string(index=False))

    # --- Heatmap: avg Sharpe by Signal × Filter (best threshold per cell) ---
    print("\nGenerating charts ...")
    pivot_sharpe = rdf.groupby(["Signal", "Filter"])["Sharpe"].max().unstack()
    fig_h, ax_h = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_sharpe, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax_h)
    ax_h.set_title("Best Sharpe by Signal × Filter (across thresholds)", fontsize=13)
    plt.tight_layout()
    fig_h.savefig(RESULTS_DIR / "combo_heatmap_sharpe.png", dpi=150)
    plt.close(fig_h)

    pivot_calmar = rdf.groupby(["Signal", "Filter"])["Calmar"].max().unstack()
    fig_h2, ax_h2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_calmar, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax_h2)
    ax_h2.set_title("Best Calmar by Signal × Filter (across thresholds)", fontsize=13)
    plt.tight_layout()
    fig_h2.savefig(RESULTS_DIR / "combo_heatmap_calmar.png", dpi=150)
    plt.close(fig_h2)

    # --- Top 10 equity curves ---
    top10_keys = rdf.head(10)[["Signal", "Filter", "Threshold"]].values.tolist()
    fig_eq, ax_eq = plt.subplots(figsize=(15, 7))
    plotted_mkt = False
    for sig_n, filt_n, thresh_n in top10_keys:
        match = [r for r in all_results if r["Signal"] == sig_n and r["Filter"] == filt_n and r["Threshold"] == thresh_n]
        if match:
            r = match[0]
            label = f"{sig_n} | {filt_n} | {thresh_n} (S={r['Sharpe']:.3f})"
            r["equity"].plot(ax=ax_eq, label=label, linewidth=1.1, alpha=0.8)
            if not plotted_mkt:
                r["equity_mkt"].plot(ax=ax_eq, label="SPY Buy & Hold", color="black", ls="--", lw=1.5)
                plotted_mkt = True

    # Baseline
    base = [r for r in all_results if r["Signal"] == "β_z×R²×β" and r["Filter"] == "None" and r["Threshold"] == "0.7/-0.7"]
    if base:
        base[0]["equity"].plot(ax=ax_eq, label=f"Baseline (S={base[0]['Sharpe']:.3f})", color="gray", ls=":", lw=2)
    ax_eq.set_title("Top 10 Combinations — Equity Curves", fontsize=14)
    ax_eq.set_ylabel("Growth of $1")
    ax_eq.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig_eq.savefig(RESULTS_DIR / "combo_top10_equity.png", dpi=150)
    plt.close(fig_eq)

    # --- Summary bar: top 10 Sharpe + MaxDD ---
    fig_bar, ax_bar = plt.subplots(figsize=(14, 7))
    top10_df = rdf.head(10).copy()
    labels_bar = [f"{r.Signal} | {r.Filter} | {r.Threshold}" for r in top10_df.itertuples()]
    x = np.arange(len(labels_bar))
    w = 0.35
    ax_bar.barh(x - w / 2, top10_df["Sharpe"].values, w, label="Sharpe", color="tab:blue", alpha=0.8)
    ax_dd = ax_bar.twiny()
    ax_dd.barh(x + w / 2, top10_df["MaxDD"].abs().values * 100, w, label="|MaxDD| %", color="tab:red", alpha=0.5)
    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(labels_bar, fontsize=7)
    ax_bar.set_xlabel("Sharpe Ratio")
    ax_dd.set_xlabel("|Max Drawdown| %")
    ax_bar.set_title("Top 10 Combos: Sharpe & Max Drawdown", fontsize=13)
    ax_bar.legend(loc="lower right", fontsize=9)
    ax_dd.legend(loc="lower center", fontsize=9)
    plt.tight_layout()
    fig_bar.savefig(RESULTS_DIR / "combo_top10_bar.png", dpi=150)
    plt.close(fig_bar)

    print(f"All charts saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
