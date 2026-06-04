#!/usr/bin/env python3
"""Generate the final equity curve chart: recommended config vs baseline vs B&H.

Usage
-----
    FMP_API_KEY=<key> python final_chart.py
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
    df_p["vol_sma_50"] = df_p["volume"].rolling(50).mean()
    return df_p


def run_backtest(df_p: pd.DataFrame, buy_t: float, sell_t: float,
                 use_r2: bool, use_vol: bool) -> pd.Series:
    signal = df_p["signal"]

    pos = []
    prev = 0
    for i, (s, r2, v, vsma) in enumerate(zip(
        signal, df_p["rsquare"], df_p["volume"], df_p["vol_sma_50"]
    )):
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

    return (1 + strat_ret.fillna(0)).cumprod()


def main() -> None:
    print("Fetching SPY data ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars")

    print("Computing indicators ...")
    df_p = compute_data(df)

    first_valid = df_p["signal"].first_valid_index()

    # 1. Recommended config: β_z×R²×β + R²>0.9 + Vol>SMA50 + 0.7/-0.8
    eq_recommended = run_backtest(df_p, 0.7, -0.8, use_r2=True, use_vol=True)

    # 2. Baseline: β_z×R²×β + no filter + 0.7/-0.7
    eq_baseline = run_backtest(df_p, 0.7, -0.7, use_r2=False, use_vol=False)

    # 3. Buy & Hold
    mkt_ret = df_p["daily_ret"].loc[first_valid:]
    eq_bh = (1 + mkt_ret.fillna(0)).cumprod()

    # Stats helper
    def stats(eq: pd.Series) -> dict:
        rets = eq.pct_change().dropna()
        n = len(rets)
        total = eq.iloc[-1] / eq.iloc[0] - 1
        ann = (1 + total) ** (252 / n) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = eq / eq.cummax() - 1
        maxdd = dd.min()
        calmar = ann / abs(maxdd) if maxdd < 0 else 0
        return {"Total": total, "Ann": ann, "Vol": vol,
                "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar}

    s_rec = stats(eq_recommended)
    s_base = stats(eq_baseline)
    s_bh = stats(eq_bh)

    print(f"\nRecommended: Sharpe={s_rec['Sharpe']:.3f}  Total={s_rec['Total']:.1%}  MaxDD={s_rec['MaxDD']:.2%}")
    print(f"Baseline:    Sharpe={s_base['Sharpe']:.3f}  Total={s_base['Total']:.1%}  MaxDD={s_base['MaxDD']:.2%}")
    print(f"Buy & Hold:  Sharpe={s_bh['Sharpe']:.3f}  Total={s_bh['Total']:.1%}  MaxDD={s_bh['MaxDD']:.2%}")

    # --- Main equity curve chart ---
    fig, ax = plt.subplots(figsize=(14, 7))

    eq_recommended.plot(ax=ax, label=(
        f"Recommended: β_z×R²×β + R²>0.9 + Vol>SMA50 + 0.7/-0.8\n"
        f"Sharpe {s_rec['Sharpe']:.3f} | Total {s_rec['Total']:.0%} | MaxDD {s_rec['MaxDD']:.1%} | Calmar {s_rec['Calmar']:.3f}"
    ), color="tab:blue", linewidth=2)

    eq_baseline.plot(ax=ax, label=(
        f"Baseline: β_z×R²×β + No filter + 0.7/-0.7\n"
        f"Sharpe {s_base['Sharpe']:.3f} | Total {s_base['Total']:.0%} | MaxDD {s_base['MaxDD']:.1%} | Calmar {s_base['Calmar']:.3f}"
    ), color="tab:orange", linewidth=1.5, alpha=0.8)

    eq_bh.plot(ax=ax, label=(
        f"SPY Buy & Hold\n"
        f"Sharpe {s_bh['Sharpe']:.3f} | Total {s_bh['Total']:.0%} | MaxDD {s_bh['MaxDD']:.1%}"
    ), color="black", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_title("RSRS Market Timing Strategy — SPY (2004-2026)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Growth of $1", fontsize=12)
    ax.set_xlabel("")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "final_equity_curve.png", dpi=150)
    plt.close(fig)

    # --- Drawdown comparison ---
    fig2, ax2 = plt.subplots(figsize=(14, 4))

    dd_rec = (eq_recommended / eq_recommended.cummax() - 1) * 100
    dd_base = (eq_baseline / eq_baseline.cummax() - 1) * 100
    dd_bh = (eq_bh / eq_bh.cummax() - 1) * 100

    dd_rec.plot(ax=ax2, label="Recommended", color="tab:blue", linewidth=1.2, alpha=0.8)
    dd_base.plot(ax=ax2, label="Baseline", color="tab:orange", linewidth=1, alpha=0.7)
    dd_bh.plot(ax=ax2, label="SPY Buy & Hold", color="black", linewidth=1, linestyle="--", alpha=0.6)

    ax2.set_title("Drawdown Comparison", fontsize=13)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "final_drawdown.png", dpi=150)
    plt.close(fig2)

    print(f"\nCharts saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
