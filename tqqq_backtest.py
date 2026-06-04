#!/usr/bin/env python3
"""Apply the recommended RSRS config to TQQQ.

Config: β_z×R²×β + R²>0.9 + Vol>SMA(vol,50) + Buy=0.7 / Sell=-0.8

TQQQ is a 3x leveraged QQQ ETF (inception ~2010-02). Because TQQQ has
shorter history than SPY, we have two approaches for the RSRS signal:

  A) "TQQQ signal": compute RSRS from TQQQ's own High/Low. Limited by
     TQQQ's ~14 years of history — the 1118-day warmup means signal
     starts ~2014.

  B) "QQQ signal → TQQQ execution": compute RSRS from QQQ's High/Low
     (longer history, same underlying), but trade TQQQ. This gives a
     longer backtest window since QQQ started in 1999.

We run both approaches and compare.

Usage
-----
    FMP_API_KEY=<key> python tqqq_backtest.py
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
BUY_T = 0.7
SELL_T = -0.8
TC_RATE = 5 / 10_000


def compute_rsrs(df_signal: pd.DataFrame) -> pd.DataFrame:
    """Compute RSRS signal + R² from a price DataFrame (must have high/low)."""
    df_p = df_signal.copy()
    rsrs = RSRSIndicator(ols_window=OLS_WINDOW, zscore_window=ZSCORE_WINDOW)
    sigs, rsqs = [], []
    for _, row in df_p.iterrows():
        rsrs.update_raw(float(row["high"]), float(row["low"]))
        sigs.append(rsrs.signal_value if rsrs.initialized else float("nan"))
        rsqs.append(rsrs.rsquare)
    df_p["signal"] = sigs
    df_p["rsquare"] = rsqs
    return df_p


def backtest(signal: pd.Series, rsq: pd.Series, volume: pd.Series,
             daily_ret: pd.Series, label: str) -> dict:
    """Run backtest with recommended config."""
    vol_sma_50 = volume.rolling(50).mean()
    rsq_gate = (rsq > 0.9).fillna(False)
    vol_gate = (volume > vol_sma_50).fillna(False)

    pos = []
    prev = 0
    for s, rg, vg in zip(signal, rsq_gate, vol_gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > BUY_T and rg and vg:
            prev = 1
            pos.append(prev)
        elif s < SELL_T:
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

    # Win rate
    trade_changes = pos_series.diff().fillna(0)
    entries = pos_series.index[trade_changes == 1]
    exits = pos_series.index[trade_changes == -1]
    wins = 0
    total_trades = 0
    for entry in entries:
        matching_exits = exits[exits > entry]
        if len(matching_exits) > 0:
            exit_date = matching_exits[0]
            trade_ret = (eq.loc[exit_date] / eq.loc[entry]) - 1
            if trade_ret > 0:
                wins += 1
            total_trades += 1
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    return {
        "label": label, "Total": total, "Ann ret": ann, "Ann vol": vol,
        "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar,
        "Trades": trades, "Win rate": win_rate,
        "equity": eq, "equity_mkt": eq_mkt,
    }


def main() -> None:
    print("Fetching data ...")
    tqqq = fetch_fmp_daily("TQQQ", start="2010-01-01")
    qqq = fetch_fmp_daily("QQQ", start="2000-01-01")
    spy = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  TQQQ: {len(tqqq)} bars ({tqqq['date'].min()} to {tqqq['date'].max()})")
    print(f"  QQQ:  {len(qqq)} bars ({qqq['date'].min()} to {qqq['date'].max()})")
    print(f"  SPY:  {len(spy)} bars ({spy['date'].min()} to {spy['date'].max()})")

    # --- Approach A: TQQQ signal from TQQQ's own H/L ---
    print("\n[A] Computing RSRS from TQQQ's own High/Low ...")
    tqqq_df = tqqq.set_index("date").sort_index()
    tqqq_rsrs = compute_rsrs(tqqq_df)
    tqqq_rsrs["daily_ret"] = tqqq_rsrs["open"].pct_change()

    res_a = backtest(
        tqqq_rsrs["signal"], tqqq_rsrs["rsquare"], tqqq_rsrs["volume"],
        tqqq_rsrs["daily_ret"], "TQQQ signal → TQQQ"
    )

    # --- Approach B: QQQ signal → TQQQ execution ---
    print("[B] Computing RSRS from QQQ's High/Low, trading TQQQ ...")
    qqq_df = qqq.set_index("date").sort_index()
    qqq_rsrs = compute_rsrs(qqq_df)

    # Align QQQ signal to TQQQ dates
    tqqq_dates = tqqq_df.index
    qqq_signal_aligned = qqq_rsrs["signal"].reindex(tqqq_dates)
    qqq_rsq_aligned = qqq_rsrs["rsquare"].reindex(tqqq_dates)

    res_b = backtest(
        qqq_signal_aligned, qqq_rsq_aligned, tqqq_rsrs["volume"],
        tqqq_rsrs["daily_ret"], "QQQ signal → TQQQ"
    )

    # --- Approach C: SPY signal → TQQQ execution ---
    print("[C] Computing RSRS from SPY's High/Low, trading TQQQ ...")
    spy_df = spy.set_index("date").sort_index()
    spy_rsrs = compute_rsrs(spy_df)

    spy_signal_aligned = spy_rsrs["signal"].reindex(tqqq_dates)
    spy_rsq_aligned = spy_rsrs["rsquare"].reindex(tqqq_dates)

    res_c = backtest(
        spy_signal_aligned, spy_rsq_aligned, tqqq_rsrs["volume"],
        tqqq_rsrs["daily_ret"], "SPY signal → TQQQ"
    )

    # --- Also run recommended config on SPY for comparison ---
    print("[D] Running same config on SPY for comparison ...")
    spy_rsrs["daily_ret"] = spy_rsrs["open"].pct_change()
    spy_rsrs["vol_sma_50"] = spy_rsrs["volume"].rolling(50).mean()

    res_spy = backtest(
        spy_rsrs["signal"], spy_rsrs["rsquare"], spy_rsrs["volume"],
        spy_rsrs["daily_ret"], "SPY signal → SPY (reference)"
    )

    # --- Summary ---
    results = [r for r in [res_a, res_b, res_c, res_spy] if r]
    print(f"\n{'='*100}")
    print("β_z×R²×β + R²>0.9 + Vol>SMA(vol,50) + 0.7/-0.8")
    print(f"{'='*100}")

    fmt = {"Total": "{:.1%}", "Ann ret": "{:.2%}", "Ann vol": "{:.2%}",
           "Sharpe": "{:.3f}", "MaxDD": "{:.2%}", "Calmar": "{:.3f}", "Win rate": "{:.1%}"}
    rows = []
    for r in results:
        row = {"Approach": r["label"]}
        for k in ["Total", "Ann ret", "Ann vol", "Sharpe", "MaxDD", "Calmar", "Trades", "Win rate"]:
            row[k] = fmt.get(k, "{}").format(r[k]) if k in fmt else r[k]
        rows.append(row)
        print(f"  {r['label']:30s}  Total={r['Total']:.1%}  Sharpe={r['Sharpe']:.3f}  MaxDD={r['MaxDD']:.2%}  Calmar={r['Calmar']:.3f}  Win={r['Win rate']:.0%}  Trades={r['Trades']}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(RESULTS_DIR / "tqqq_results.csv", index=False)

    # --- Buy & Hold stats ---
    first_valid_b = qqq_signal_aligned.first_valid_index()
    if first_valid_b and first_valid_b in tqqq_rsrs.index:
        bh_ret = tqqq_rsrs["daily_ret"].loc[first_valid_b:]
        bh_eq = (1 + bh_ret.fillna(0)).cumprod()
        bh_total = bh_eq.iloc[-1] / bh_eq.iloc[0] - 1
        bh_dd = (bh_eq / bh_eq.cummax() - 1).min()
        bh_rets = bh_eq.pct_change().dropna()
        bh_vol = bh_rets.std() * np.sqrt(252)
        bh_ann = (1 + bh_total) ** (252 / len(bh_rets)) - 1
        bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
        print(f"\n  TQQQ Buy & Hold:                  Total={bh_total:.1%}  Sharpe={bh_sharpe:.3f}  MaxDD={bh_dd:.2%}")

    # --- Charts ---
    print("\nGenerating charts ...")

    # 1. Equity curves comparison
    fig1, ax1 = plt.subplots(figsize=(15, 7))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:gray"]
    for i, r in enumerate(results):
        r["equity"].plot(ax=ax1, label=f"{r['label']} (S={r['Sharpe']:.3f})", color=colors[i], lw=1.5)
    # TQQQ buy & hold
    if res_b:
        res_b["equity_mkt"].plot(ax=ax1, label="TQQQ Buy & Hold", color="black", ls="--", lw=1.5)
    ax1.set_title("RSRS on TQQQ — Equity Curves", fontsize=14)
    ax1.set_ylabel("Growth of $1")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / "tqqq_equity.png", dpi=150)
    plt.close(fig1)

    # 2. Drawdown comparison
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    for i, r in enumerate(results[:3]):
        dd = (r["equity"] / r["equity"].cummax() - 1) * 100
        dd.plot(ax=ax2, label=r["label"], color=colors[i], lw=1, alpha=0.8)
    if res_b:
        dd_bh = (res_b["equity_mkt"] / res_b["equity_mkt"].cummax() - 1) * 100
        dd_bh.plot(ax=ax2, label="TQQQ Buy & Hold", color="black", ls="--", lw=1.5)
    ax2.set_title("Drawdown: RSRS-Timed TQQQ vs Buy & Hold", fontsize=13)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "tqqq_drawdown.png", dpi=150)
    plt.close(fig2)

    # 3. Bar chart comparison
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))
    labels = [r["label"] for r in results]
    sharpes = [r["Sharpe"] for r in results]
    maxdds = [abs(r["MaxDD"]) * 100 for r in results]

    x = np.arange(len(labels))
    ax3a.bar(x, sharpes, color=colors[:len(results)], alpha=0.8)
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax3a.set_ylabel("Sharpe Ratio")
    ax3a.set_title("Sharpe Ratio by Approach")

    ax3b.bar(x, maxdds, color=colors[:len(results)], alpha=0.8)
    ax3b.set_xticks(x)
    ax3b.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax3b.set_ylabel("|Max Drawdown| %")
    ax3b.set_title("Max Drawdown by Approach")

    plt.suptitle("RSRS Config on TQQQ — Performance Comparison", fontsize=14)
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / "tqqq_comparison.png", dpi=150)
    plt.close(fig3)

    print(f"Charts saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
