#!/usr/bin/env python3
"""EWT (Taiwan Market) parameter localization.

Optimize RSRS parameters specifically for the Taiwan market (EWT ETF):
- OLS window N: 10, 14, 18, 22, 26
- Z-score window M: 500, 700, 900, 1100
- Buy/Sell thresholds: multiple asymmetric combos
- Filters: R², Volume, combinations

Usage
-----
    FMP_API_KEY=<key> python ewt_optimization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import product

from data_loader import fetch_fmp_daily

sns.set_theme(style="whitegrid")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

TC_RATE = 5 / 10_000


def compute_rsrs_custom(df: pd.DataFrame, ols_window: int, zscore_window: int) -> pd.DataFrame:
    """Compute RSRS with custom N/M parameters."""
    from numpy.linalg import lstsq

    df_p = df.set_index("date").sort_index().copy()
    highs = df_p["high"].values
    lows = df_p["low"].values
    n = len(df_p)

    betas = np.full(n, np.nan)
    rsquares = np.full(n, np.nan)

    for i in range(ols_window - 1, n):
        y = highs[i - ols_window + 1:i + 1]
        x = lows[i - ols_window + 1:i + 1]
        X = np.column_stack([x, np.ones(ols_window)])
        coef, residuals, _, _ = lstsq(X, y, rcond=None)
        betas[i] = coef[0]
        # R²
        y_hat = X @ coef
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        rsquares[i] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Z-score
    beta_z = np.full(n, np.nan)
    for i in range(ols_window + zscore_window - 2, n):
        window = betas[i - zscore_window + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= 50:
            mu = np.mean(valid)
            std = np.std(valid)
            if std > 0:
                beta_z[i] = (betas[i] - mu) / std

    df_p["beta"] = betas
    df_p["beta_z"] = beta_z
    df_p["rsquare"] = rsquares
    # Right-skew signal: beta_z × R² × beta
    df_p["signal"] = beta_z * rsquares * betas
    df_p["daily_ret"] = df_p["open"].pct_change()
    df_p["vol_sma_50"] = df_p["volume"].rolling(50).mean()
    df_p["vol_sma_20"] = df_p["volume"].rolling(20).mean()

    return df_p


def backtest_ewt(df_p: pd.DataFrame, buy_t: float, sell_t: float,
                 r2_threshold: float | None = None,
                 vol_filter: str | None = None) -> dict:
    """Run backtest with given parameters."""
    signal = df_p["signal"]
    pos = []
    prev = 0
    for i, (s, r2, v, vsma50, vsma20) in enumerate(zip(
        signal, df_p["rsquare"], df_p["volume"],
        df_p["vol_sma_50"], df_p["vol_sma_20"]
    )):
        if pd.isna(s):
            pos.append(prev)
            continue
        gate = True
        if r2_threshold is not None and r2 <= r2_threshold:
            gate = False
        if vol_filter == "sma50" and (pd.isna(vsma50) or v <= vsma50):
            gate = False
        elif vol_filter == "sma20" and (pd.isna(vsma20) or v <= vsma20):
            gate = False
        if s > buy_t and gate:
            prev = 1
        elif s < sell_t:
            prev = 0
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

    eq = (1 + strat_ret.fillna(0)).cumprod()
    rets = eq.pct_change().dropna()
    n_days = len(rets)
    if n_days < 50:
        return {}

    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (252 / n_days) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else 0
    trades = int((pos_series.diff().abs() > 0).sum())

    return {
        "Total": total, "Ann": ann, "Vol": vol,
        "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar,
        "Trades": trades, "equity": eq,
    }


def main() -> None:
    print("=" * 80)
    print("EWT Parameter Localization — Optimizing RSRS for Taiwan Market")
    print("=" * 80)

    print("\nFetching EWT data ...")
    ewt = fetch_fmp_daily("EWT", start="2000-01-01")
    print(f"  EWT: {len(ewt)} bars ({ewt['date'].min()} to {ewt['date'].max()})")

    # --- Phase 1: OLS Window (N) and Z-score Window (M) ---
    print("\n--- Phase 1: N/M Window Optimization ---")
    n_values = [10, 14, 18, 22, 26]
    m_values = [500, 700, 900, 1100]

    nm_results = []
    for n_val, m_val in product(n_values, m_values):
        df_p = compute_rsrs_custom(ewt, ols_window=n_val, zscore_window=m_val)
        res = backtest_ewt(df_p, buy_t=0.7, sell_t=-0.7)
        if res:
            nm_results.append({
                "N": n_val, "M": m_val,
                "Sharpe": res["Sharpe"], "Total": res["Total"],
                "MaxDD": res["MaxDD"], "Calmar": res["Calmar"],
                "Trades": res["Trades"],
            })
            print(f"  N={n_val:2d} M={m_val:4d}: Sharpe={res['Sharpe']:.3f} "
                  f"Total={res['Total']:.1%} MaxDD={res['MaxDD']:.2%}")

    nm_df = pd.DataFrame(nm_results).sort_values("Sharpe", ascending=False)
    print(f"\n  Best N/M: N={nm_df.iloc[0]['N']:.0f}, M={nm_df.iloc[0]['M']:.0f} "
          f"(Sharpe={nm_df.iloc[0]['Sharpe']:.3f})")

    best_n = int(nm_df.iloc[0]["N"])
    best_m = int(nm_df.iloc[0]["M"])

    # --- Phase 2: Threshold optimization with best N/M ---
    print(f"\n--- Phase 2: Threshold Optimization (N={best_n}, M={best_m}) ---")
    buy_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sell_values = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]

    df_p = compute_rsrs_custom(ewt, ols_window=best_n, zscore_window=best_m)
    thresh_results = []
    for buy_t, sell_t in product(buy_values, sell_values):
        res = backtest_ewt(df_p, buy_t=buy_t, sell_t=sell_t)
        if res:
            thresh_results.append({
                "Buy": buy_t, "Sell": sell_t,
                "Sharpe": res["Sharpe"], "Total": res["Total"],
                "MaxDD": res["MaxDD"], "Calmar": res["Calmar"],
                "Trades": res["Trades"],
            })

    thresh_df = pd.DataFrame(thresh_results).sort_values("Sharpe", ascending=False)
    print("\n  Top 5 Thresholds:")
    for _, row in thresh_df.head(5).iterrows():
        print(f"    Buy={row['Buy']:.1f} Sell={row['Sell']:.1f}: "
              f"Sharpe={row['Sharpe']:.3f} Total={row['Total']:.1%} "
              f"MaxDD={row['MaxDD']:.2%} Calmar={row['Calmar']:.3f}")

    best_buy = thresh_df.iloc[0]["Buy"]
    best_sell = thresh_df.iloc[0]["Sell"]

    # --- Phase 3: Filter optimization ---
    print(f"\n--- Phase 3: Filter Optimization (N={best_n}, M={best_m}, "
          f"Buy={best_buy}, Sell={best_sell}) ---")

    filter_configs = [
        ("No filter", None, None),
        ("R²>0.8", 0.8, None),
        ("R²>0.85", 0.85, None),
        ("R²>0.9", 0.9, None),
        ("Vol>SMA50", None, "sma50"),
        ("Vol>SMA20", None, "sma20"),
        ("R²>0.8 + Vol>SMA50", 0.8, "sma50"),
        ("R²>0.85 + Vol>SMA50", 0.85, "sma50"),
        ("R²>0.9 + Vol>SMA50", 0.9, "sma50"),
        ("R²>0.8 + Vol>SMA20", 0.8, "sma20"),
    ]

    filter_results = []
    for label, r2_t, vol_f in filter_configs:
        res = backtest_ewt(df_p, buy_t=best_buy, sell_t=best_sell,
                           r2_threshold=r2_t, vol_filter=vol_f)
        if res:
            filter_results.append({
                "Filter": label, "Sharpe": res["Sharpe"],
                "Total": res["Total"], "MaxDD": res["MaxDD"],
                "Calmar": res["Calmar"], "Trades": res["Trades"],
            })
            print(f"  {label:<25}: Sharpe={res['Sharpe']:.3f} "
                  f"Total={res['Total']:.1%} MaxDD={res['MaxDD']:.2%} "
                  f"Calmar={res['Calmar']:.3f}")

    filter_df = pd.DataFrame(filter_results).sort_values("Sharpe", ascending=False)

    # --- Final: Best EWT config ---
    best_filter_row = filter_df.iloc[0]
    best_filter_label = best_filter_row["Filter"]

    # Get best filter params
    best_r2 = None
    best_vol = None
    for label, r2_t, vol_f in filter_configs:
        if label == best_filter_label:
            best_r2 = r2_t
            best_vol = vol_f
            break

    # Run final backtest and get equity
    final_res = backtest_ewt(df_p, buy_t=best_buy, sell_t=best_sell,
                             r2_threshold=best_r2, vol_filter=best_vol)
    # Also run SPY default for comparison
    baseline_res = backtest_ewt(df_p, buy_t=0.7, sell_t=-0.7)

    print("\n" + "=" * 80)
    print("FINAL OPTIMIZED EWT CONFIGURATION")
    print("=" * 80)
    print(f"  OLS Window N:     {best_n}")
    print(f"  Z-score Window M: {best_m}")
    print(f"  Buy Threshold:    {best_buy}")
    print(f"  Sell Threshold:   {best_sell}")
    print(f"  Filter:           {best_filter_label}")
    print("  ---")
    if final_res:
        print(f"  Sharpe:           {final_res['Sharpe']:.3f}")
        print(f"  Total Return:     {final_res['Total']:.1%}")
        print(f"  Max Drawdown:     {final_res['MaxDD']:.2%}")
        print(f"  Calmar:           {final_res['Calmar']:.3f}")
        print(f"  Trades:           {final_res['Trades']}")

    # --- Generate comparison chart ---
    fig, ax = plt.subplots(figsize=(14, 7))
    if final_res and "equity" in final_res:
        final_res["equity"].plot(ax=ax, label=(
            f"EWT Optimized (N={best_n},M={best_m},{best_buy}/{best_sell},"
            f"{best_filter_label}) S={final_res['Sharpe']:.3f}"
        ), color="tab:blue", lw=2)
    if baseline_res and "equity" in baseline_res:
        baseline_res["equity"].plot(ax=ax, label=(
            f"EWT Default SPY Params (N=18,M=1100,0.7/-0.7) "
            f"S={baseline_res['Sharpe']:.3f}"
        ), color="tab:orange", lw=1.5, alpha=0.8)

    # B&H
    first_valid = df_p["signal"].first_valid_index()
    if first_valid is not None:
        mkt_ret = df_p["daily_ret"].loc[first_valid:]
        eq_mkt = (1 + mkt_ret.fillna(0)).cumprod()
        eq_mkt.plot(ax=ax, label="EWT Buy & Hold", color="black", ls="--", lw=1.5, alpha=0.7)

    ax.set_title("EWT (Taiwan Market) — Localized RSRS Parameters vs Default", fontsize=14)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "ewt_optimized.png", dpi=150)
    plt.close(fig)
    print("\n  Saved ewt_optimized.png")

    # Heatmap for thresholds
    pivot = thresh_df.pivot_table(index="Sell", columns="Buy", values="Sharpe")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(f"EWT Threshold Heatmap (N={best_n}, M={best_m}) — Sharpe", fontsize=13)
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "ewt_threshold_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved ewt_threshold_heatmap.png")

    # Save all results
    nm_df.to_csv(RESULTS_DIR / "ewt_nm_search.csv", index=False)
    thresh_df.to_csv(RESULTS_DIR / "ewt_threshold_search.csv", index=False)
    filter_df.to_csv(RESULTS_DIR / "ewt_filter_search.csv", index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
