#!/usr/bin/env python3
"""Cross-market validation: apply RSRS to EWT (iShares MSCI Taiwan ETF).

Also generates crisis-period zoom-in charts (2008, 2020) with buy/sell signals.

Usage
-----
    FMP_API_KEY=<key> python cross_market_validation.py
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
DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

OLS_WINDOW = 18
ZSCORE_WINDOW = 1100
BUY_T = 0.7
SELL_T = -0.8
TC_RATE = 5 / 10_000


def compute_rsrs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSRS signal + R² from a price DataFrame."""
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
    df_p["vol_sma_50"] = df_p["volume"].rolling(50).mean()
    return df_p


def backtest(df_p: pd.DataFrame, buy_t: float, sell_t: float,
             use_r2: bool = True, use_vol: bool = True) -> dict:
    """Run backtest with recommended config."""
    signal = df_p["signal"]
    pos = []
    prev = 0
    for s, r2, v, vsma in zip(
        signal, df_p["rsquare"], df_p["volume"], df_p["vol_sma_50"]
    ):
        if pd.isna(s):
            pos.append(prev)
            continue
        gate = True
        if use_r2 and r2 <= 0.9:
            gate = False
        if use_vol and (pd.isna(vsma) or v <= vsma):
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
    sharpe = ann / vol if vol > 0 else 0
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else 0
    trades = int((pos_series.diff().abs() > 0).sum())

    return {
        "Total": total, "Ann ret": ann, "Ann vol": vol,
        "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar,
        "Trades": trades, "equity": eq, "equity_mkt": eq_mkt,
        "position": pos_series,
    }


def crisis_zoom_chart(df_p: pd.DataFrame, position: pd.Series,
                      start: str, end: str, title: str, fname: str) -> None:
    """Generate zoomed-in crisis chart with buy/sell signal markers."""
    mask = (df_p.index >= start) & (df_p.index <= end)
    df_zoom = df_p.loc[mask]
    pos_zoom = position.loc[mask]

    if len(df_zoom) == 0:
        return

    # Detect buy/sell points
    pos_diff = pos_zoom.diff()
    buy_dates = pos_zoom.index[pos_diff == 1]
    sell_dates = pos_zoom.index[pos_diff == -1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                   sharex=True)

    # Price chart
    ax1.plot(df_zoom.index, df_zoom["close"], color="black", lw=1.2, label="Close")

    # Shade holding periods
    in_position = False
    entry_date = None
    for i in range(len(pos_zoom)):
        if pos_zoom.iloc[i] == 1 and not in_position:
            in_position = True
            entry_date = pos_zoom.index[i]
        elif pos_zoom.iloc[i] == 0 and in_position:
            in_position = False
            ax1.axvspan(entry_date, pos_zoom.index[i], alpha=0.15, color="green")
    if in_position and entry_date is not None:
        ax1.axvspan(entry_date, pos_zoom.index[-1], alpha=0.15, color="green")

    # Buy/Sell markers
    for bd in buy_dates:
        if bd in df_zoom.index:
            ax1.scatter(bd, df_zoom.loc[bd, "close"], marker="^", s=120,
                        color="green", zorder=5)
    for sd in sell_dates:
        if sd in df_zoom.index:
            ax1.scatter(sd, df_zoom.loc[sd, "close"], marker="v", s=120,
                        color="red", zorder=5)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.legend(["Close", "Buy Signal", "Sell Signal"], loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Signal chart
    sig_zoom = df_p["signal"].loc[mask]
    ax2.plot(sig_zoom.index, sig_zoom, color="tab:blue", lw=1)
    ax2.axhline(BUY_T, color="green", ls="--", lw=0.8, label=f"Buy={BUY_T}")
    ax2.axhline(SELL_T, color="red", ls="--", lw=0.8, label=f"Sell={SELL_T}")
    ax2.axhline(0, color="gray", ls="-", lw=0.5)
    ax2.set_ylabel("RSRS Signal")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(DOCS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def main() -> None:
    # --- Part 1: Cross-market validation (EWT = Taiwan) ---
    print("=" * 80)
    print("Part 1: Cross-Market Validation — EWT (iShares MSCI Taiwan ETF)")
    print("=" * 80)

    print("\nFetching data ...")
    ewt = fetch_fmp_daily("EWT", start="2000-01-01")
    spy = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  EWT: {len(ewt)} bars ({ewt['date'].min()} to {ewt['date'].max()})")
    print(f"  SPY: {len(spy)} bars ({spy['date'].min()} to {spy['date'].max()})")

    print("\nComputing RSRS for EWT ...")
    ewt_rsrs = compute_rsrs(ewt)

    print("Computing RSRS for SPY ...")
    spy_rsrs = compute_rsrs(spy)

    # Run backtest on EWT — recommended config
    print("\nRunning backtest on EWT (recommended config) ...")
    res_ewt = backtest(ewt_rsrs, BUY_T, SELL_T, use_r2=True, use_vol=True)

    # EWT with no filter (baseline)
    res_ewt_base = backtest(ewt_rsrs, 0.7, -0.7, use_r2=False, use_vol=False)

    # SPY reference
    res_spy = backtest(spy_rsrs, BUY_T, SELL_T, use_r2=True, use_vol=True)

    print("\n--- Results ---")
    print(f"{'Config':<45} {'Sharpe':>7} {'Total':>8} {'MaxDD':>8} {'Calmar':>7} {'Trades':>6}")
    print("-" * 90)
    for label, r in [
        ("EWT Recommended (R2+Vol+0.7/-0.8)", res_ewt),
        ("EWT Baseline (no filter, 0.7/-0.7)", res_ewt_base),
        ("SPY Recommended (reference)", res_spy),
    ]:
        if r:
            print(f"  {label:<43} {r['Sharpe']:>7.3f} {r['Total']:>7.1%} "
                  f"{r['MaxDD']:>7.2%} {r['Calmar']:>7.3f} {r['Trades']:>6}")

    # EWT equity chart
    fig, ax = plt.subplots(figsize=(14, 7))
    if res_ewt:
        res_ewt["equity"].plot(ax=ax, label=(
            f"EWT RSRS Recommended (S={res_ewt['Sharpe']:.3f}, "
            f"DD={res_ewt['MaxDD']:.1%})"
        ), color="tab:blue", lw=2)
    if res_ewt_base:
        res_ewt_base["equity"].plot(ax=ax, label=(
            f"EWT RSRS Baseline (S={res_ewt_base['Sharpe']:.3f}, "
            f"DD={res_ewt_base['MaxDD']:.1%})"
        ), color="tab:orange", lw=1.5, alpha=0.8)
    if res_ewt:
        res_ewt["equity_mkt"].plot(ax=ax, label="EWT Buy & Hold",
                                   color="black", ls="--", lw=1.5, alpha=0.7)
    ax.set_title("RSRS Cross-Market Validation: EWT (Taiwan Market ETF)", fontsize=14)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(DOCS_DIR / "ewt_equity.png", dpi=150)
    plt.close(fig)
    print("\n  Saved ewt_equity.png")

    # Save results CSV
    rows = []
    for label, r in [
        ("EWT Recommended", res_ewt),
        ("EWT Baseline", res_ewt_base),
        ("SPY Recommended", res_spy),
    ]:
        if r:
            rows.append({
                "Market": label,
                "Total": f"{r['Total']:.1%}",
                "Sharpe": f"{r['Sharpe']:.3f}",
                "MaxDD": f"{r['MaxDD']:.2%}",
                "Calmar": f"{r['Calmar']:.3f}",
                "Trades": r["Trades"],
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "cross_market.csv", index=False)

    # --- Part 2: Crisis zoom-in charts (SPY) ---
    print("\n" + "=" * 80)
    print("Part 2: Crisis Period Zoom-In — Buy/Sell Signal Visualization")
    print("=" * 80)

    pos_spy = res_spy["position"] if res_spy else pd.Series(dtype=float)

    # 2008 Financial Crisis
    crisis_zoom_chart(
        spy_rsrs, pos_spy,
        "2007-10-01", "2009-06-30",
        "2008 Financial Crisis — RSRS Buy/Sell Signals on SPY",
        "crisis_2008_zoom.png"
    )

    # 2020 COVID Crash
    crisis_zoom_chart(
        spy_rsrs, pos_spy,
        "2020-01-01", "2020-12-31",
        "2020 COVID Crash — RSRS Buy/Sell Signals on SPY",
        "crisis_2020_zoom.png"
    )

    # 2022 Bear Market
    crisis_zoom_chart(
        spy_rsrs, pos_spy,
        "2022-01-01", "2023-01-31",
        "2022 Bear Market — RSRS Buy/Sell Signals on SPY",
        "crisis_2022_zoom.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
