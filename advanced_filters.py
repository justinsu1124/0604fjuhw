#!/usr/bin/env python3
"""Advanced filters & signal variant analysis for RSRS strategy.

Tests each filter/variant INDEPENDENTLY against the no-filter baseline:
  Part A — Filters (applied to the right-skew signal):
    1. Volatility (ATR): skip entry when ATR_20 > 1.5× rolling median
    2. Volatility (realized): skip when 20d realized vol > 1.5× 252d median
    3. Trend SMA200: only buy when close > SMA(200)
    4. Trend MA slope: only buy when SMA(50) is rising (slope > 0 over 10d)
    5. RSI filter: skip entry when RSI(14) > 70 (overbought)
    6. R² gate (>0.8): only trust signal when R² > 0.8
    7. R² gate (>0.9): only trust signal when R² > 0.9

  Part B — Signal variants (each with buy=0.7, sell=-0.7):
    1. Standard score: beta_z
    2. R²-adjusted: beta_z × R²
    3. Right-skew (baseline): beta_z × R² × β
    4. t-value adjusted: t × beta_z / mean(t)
    5. t-value right-skew: t × beta_z × β / mean(t)

  Part C — Factor binning: forward 10-day return by signal bucket

Usage
-----
    FMP_API_KEY=<key> python advanced_filters.py
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
BUY = 0.7
SELL = -0.7
TC_RATE = 5 / 10_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Run RSRS indicator and attach all signal variants + technical helpers."""
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
            "beta": rsrs.beta,
            "tvalue": rsrs.tvalue,
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

    # Realized vol (20d)
    df_p["rvol_20"] = df_p["close"].pct_change().rolling(20).std() * np.sqrt(252)
    df_p["rvol_median"] = df_p["rvol_20"].rolling(252).median()

    # SMA
    df_p["sma_50"] = df_p["close"].rolling(50).mean()
    df_p["sma_200"] = df_p["close"].rolling(200).mean()
    df_p["sma50_slope"] = df_p["sma_50"] - df_p["sma_50"].shift(10)

    # RSI(14)
    delta = df_p["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-12)
    df_p["rsi_14"] = 100 - 100 / (1 + rs)

    return df_p


def backtest(signal: pd.Series, buy_gate: pd.Series, daily_ret: pd.Series) -> dict:
    """Run a single backtest. Returns stats dict + equity series."""
    pos = []
    prev = 0
    for s, gate in zip(signal, buy_gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > BUY and gate:
            prev = 1
            pos.append(prev)
        elif s < SELL:
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
        return {"equity": eq, "equity_mkt": eq_mkt}
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


def print_table(results: list[dict]) -> pd.DataFrame:
    """Print and return a summary table."""
    rows = []
    for r in results:
        if "Sharpe" in r:
            rows.append({
                "Variant": r["label"],
                "Total": f"{r['Total']:.1%}",
                "Ann ret": f"{r['Ann ret']:.2%}",
                "Ann vol": f"{r['Ann vol']:.2%}",
                "Sharpe": f"{r['Sharpe']:.3f}",
                "MaxDD": f"{r['MaxDD']:.2%}",
                "Calmar": f"{r['Calmar']:.3f}",
                "Trades": r["Trades"],
            })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    return tbl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Fetching SPY data from FMP ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars: {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")

    print("Computing RSRS signals & technical indicators ...")
    df_p = compute_all(df)

    always_true = pd.Series(True, index=df_p.index)
    sig = df_p["sig_right_skew"]  # baseline signal

    # ===================================================================
    # PART A: Filters on right-skew signal
    # ===================================================================
    print("\n" + "=" * 90)
    print("PART A: Advanced Filters (right-skew signal, buy=0.7 sell=-0.7)")
    print("=" * 90)

    filter_defs: list[tuple[str, pd.Series]] = [
        ("No filter (baseline)", always_true),
        ("ATR filter: ATR20 < 1.5× median", df_p["atr_20"] < 1.5 * df_p["atr_20_median"]),
        ("ATR filter: ATR20 < 2.0× median", df_p["atr_20"] < 2.0 * df_p["atr_20_median"]),
        ("Realized vol < 1.5× median", df_p["rvol_20"] < 1.5 * df_p["rvol_median"]),
        ("Realized vol < 2.0× median", df_p["rvol_20"] < 2.0 * df_p["rvol_median"]),
        ("Trend: Close > SMA(200)", df_p["close"] > df_p["sma_200"]),
        ("Trend: SMA(50) slope > 0", df_p["sma50_slope"] > 0),
        ("Trend: Close>SMA200 & slope>0", (df_p["close"] > df_p["sma_200"]) & (df_p["sma50_slope"] > 0)),
        ("RSI(14) < 70", df_p["rsi_14"] < 70),
        ("RSI(14) < 75", df_p["rsi_14"] < 75),
        ("R² > 0.8", df_p["rsquare"] > 0.8),
        ("R² > 0.9", df_p["rsquare"] > 0.9),
        ("R² > 0.8 & ATR<1.5×", (df_p["rsquare"] > 0.8) & (df_p["atr_20"] < 1.5 * df_p["atr_20_median"])),
    ]

    filter_results = []
    for label, gate in filter_defs:
        gate = gate.fillna(False)
        res = backtest(sig, gate, df_p["daily_ret"])
        res["label"] = label
        filter_results.append(res)

    tbl_a = print_table(filter_results)
    tbl_a.to_csv(RESULTS_DIR / "advanced_filters.csv", index=False)

    # Chart A: equity curves
    fig_a, ax_a = plt.subplots(figsize=(14, 7))
    for r in filter_results:
        if "equity" in r:
            r["equity"].plot(ax=ax_a, label=r["label"], linewidth=1.1, alpha=0.8)
    if "equity_mkt" in filter_results[0]:
        filter_results[0]["equity_mkt"].plot(ax=ax_a, label="SPY Buy & Hold", color="black", ls="--", lw=1.5)
    ax_a.set_title("Part A: Advanced Filters — Equity Curves", fontsize=14)
    ax_a.set_ylabel("Growth of $1")
    ax_a.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig_a.savefig(RESULTS_DIR / "advanced_filters_equity.png", dpi=150)
    plt.close(fig_a)

    # Chart A2: Sharpe bar
    fig_a2, ax_a2 = plt.subplots(figsize=(12, 6))
    labels_a = [r["label"] for r in filter_results if "Sharpe" in r]
    sharpes_a = [r["Sharpe"] for r in filter_results if "Sharpe" in r]
    maxdds_a = [abs(r["MaxDD"]) * 100 for r in filter_results if "MaxDD" in r]
    x = np.arange(len(labels_a))
    w = 0.35
    ax_a2.barh(x - w / 2, sharpes_a, w, label="Sharpe", color="tab:blue", alpha=0.8)
    ax_a2_r = ax_a2.twiny()
    ax_a2_r.barh(x + w / 2, maxdds_a, w, label="|MaxDD| %", color="tab:red", alpha=0.5)
    ax_a2.set_yticks(x)
    ax_a2.set_yticklabels(labels_a, fontsize=8)
    ax_a2.set_xlabel("Sharpe Ratio")
    ax_a2_r.set_xlabel("|Max Drawdown| %")
    ax_a2.set_title("Part A: Sharpe & Max Drawdown by Filter", fontsize=13)
    ax_a2.legend(loc="lower right")
    ax_a2_r.legend(loc="lower center")
    plt.tight_layout()
    fig_a2.savefig(RESULTS_DIR / "advanced_filters_sharpe.png", dpi=150)
    plt.close(fig_a2)

    # ===================================================================
    # PART B: Signal variants (no filter)
    # ===================================================================
    print("\n" + "=" * 90)
    print("PART B: Signal Variants (no filter, buy=0.7 sell=-0.7)")
    print("=" * 90)

    signal_defs: list[tuple[str, pd.Series]] = [
        ("Standard score (β_z)", df_p["sig_standard"]),
        ("R²-adjusted (β_z × R²)", df_p["sig_r2adj"]),
        ("Right-skew (β_z × R² × β)", df_p["sig_right_skew"]),
        ("t-value adjusted (t × β_z / mean_t)", df_p["sig_tval_adj"]),
        ("t-value right-skew (t × β_z × β / mean_t)", df_p["sig_tval_right"]),
    ]

    signal_results = []
    for label, sig_var in signal_defs:
        res = backtest(sig_var, always_true, df_p["daily_ret"])
        res["label"] = label
        signal_results.append(res)

    tbl_b = print_table(signal_results)
    tbl_b.to_csv(RESULTS_DIR / "signal_variants.csv", index=False)

    # Chart B: equity curves
    fig_b, ax_b = plt.subplots(figsize=(14, 6))
    for r in signal_results:
        if "equity" in r:
            r["equity"].plot(ax=ax_b, label=r["label"], linewidth=1.3, alpha=0.85)
    if "equity_mkt" in signal_results[0]:
        signal_results[0]["equity_mkt"].plot(ax=ax_b, label="SPY Buy & Hold", color="black", ls="--", lw=1.5)
    ax_b.set_title("Part B: Signal Variants — Equity Curves", fontsize=14)
    ax_b.set_ylabel("Growth of $1")
    ax_b.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig_b.savefig(RESULTS_DIR / "signal_variants_equity.png", dpi=150)
    plt.close(fig_b)

    # ===================================================================
    # PART C: Factor binning (forward 10-day return by signal bucket)
    # ===================================================================
    print("\n" + "=" * 90)
    print("PART C: Factor Binning (forward 10-day return by signal bucket)")
    print("=" * 90)

    fwd_ret_10 = df_p["open"].pct_change(10).shift(-12)

    fig_c, axes_c = plt.subplots(3, 2, figsize=(16, 14))
    axes_flat = axes_c.flatten()

    for idx, (label, sig_col) in enumerate(signal_defs):
        ax = axes_flat[idx]
        combo = pd.DataFrame({"signal": sig_col, "ret10": fwd_ret_10}).dropna()
        combo["bucket"] = combo["signal"].round(1)
        means = combo.groupby("bucket")["ret10"].mean()
        counts = combo.groupby("bucket")["ret10"].count()
        # Filter buckets with at least 10 observations
        valid = counts[counts >= 10].index
        means = means.loc[valid]
        colors = ["tab:green" if v > 0 else "tab:red" for v in means.values]
        means.plot(kind="bar", ax=ax, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Mean 10d fwd return")
        ax.set_xlabel("Signal bucket")
        ax.tick_params(axis="x", labelsize=7)

    # Hide unused subplot
    if len(signal_defs) < len(axes_flat):
        for i in range(len(signal_defs), len(axes_flat)):
            axes_flat[i].set_visible(False)

    plt.suptitle("Factor Binning: Mean 10-day Forward Return by Signal Bucket", fontsize=14, y=1.01)
    plt.tight_layout()
    fig_c.savefig(RESULTS_DIR / "factor_binning.png", dpi=150, bbox_inches="tight")
    plt.close(fig_c)
    print("Factor binning chart saved.")

    print(f"\nAll results saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
