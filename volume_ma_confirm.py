#!/usr/bin/env python3
"""Volume / MA confirmation filters on the tuned strategy.

Base: β_z×R²×β + R²>0.9 + Buy=0.7 / Sell=-0.8

Tests many Volume/MA confirmation methods individually and in combinations:

Single filters:
  Volume:
    1.  Vol > SMA(vol, 5)    — very short-term volume surge
    2.  Vol > SMA(vol, 10)
    3.  Vol > SMA(vol, 20)
    4.  Vol > SMA(vol, 50)
    5.  Vol > EMA(vol, 10)
    6.  Vol > EMA(vol, 20)
    7.  Vol z-score > 0      — volume above its 60d rolling mean in std terms
    8.  Vol z-score > 0.5
    9.  Vol ratio > 1.2      — vol / SMA(vol,20) > 1.2
    10. Vol ratio > 1.5
    11. OBV slope > 0        — On-Balance Volume trending up (20d)
    12. VWAP > close         — volume-weighted avg price support

  Moving Average:
    13. Close > SMA(10)
    14. Close > SMA(20)
    15. Close > SMA(50)
    16. Close > EMA(10)
    17. Close > EMA(20)
    18. Close > EMA(50)
    19. SMA(10) > SMA(50)    — golden cross short/long
    20. SMA(20) > SMA(50)
    21. EMA(12) > EMA(26)    — MACD-like
    22. SMA(50) slope > 0    — 50d MA rising (10d lookback)
    23. SMA(20) slope > 0
    24. MACD histogram > 0

Combined (best single filters):
    25+. Top volume × Top MA combos

Usage
-----
    FMP_API_KEY=<key> python volume_ma_confirm.py
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

    v = df_p["volume"]
    c = df_p["close"]

    # --- Volume indicators ---
    for w in [5, 10, 20, 50]:
        df_p[f"vol_sma_{w}"] = v.rolling(w).mean()
    for w in [10, 20]:
        df_p[f"vol_ema_{w}"] = v.ewm(span=w, adjust=False).mean()

    # Volume z-score (60d)
    vol_mean60 = v.rolling(60).mean()
    vol_std60 = v.rolling(60).std()
    df_p["vol_zscore"] = (v - vol_mean60) / vol_std60.replace(0, 1e-12)

    # Volume ratio
    df_p["vol_ratio_20"] = v / df_p["vol_sma_20"].replace(0, 1e-12)

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df_p["obv_slope_20"] = obv - obv.shift(20)

    # VWAP (rolling 20d approximation)
    tp = (df_p["high"] + df_p["low"] + c) / 3
    df_p["vwap_20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()

    # --- Price MA indicators ---
    for w in [10, 20, 50]:
        df_p[f"sma_{w}"] = c.rolling(w).mean()
    for w in [10, 20, 50]:
        df_p[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()
    df_p["ema_12"] = c.ewm(span=12, adjust=False).mean()
    df_p["ema_26"] = c.ewm(span=26, adjust=False).mean()

    # MA slopes
    df_p["sma20_slope"] = df_p["sma_20"] - df_p["sma_20"].shift(10)
    df_p["sma50_slope"] = df_p["sma_50"] - df_p["sma_50"].shift(10)

    # MACD histogram
    macd_line = df_p["ema_12"] - df_p["ema_26"]
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    df_p["macd_hist"] = macd_line - macd_signal

    return df_p


def backtest(signal: pd.Series, rsq_gate: pd.Series, extra_gate: pd.Series,
             daily_ret: pd.Series) -> dict:
    combined_gate = rsq_gate & extra_gate

    pos = []
    prev = 0
    for s, g in zip(signal, combined_gate):
        if pd.isna(s):
            pos.append(prev)
        elif s > BUY_T and g:
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

    return {
        "Total": total, "Ann ret": ann, "Ann vol": vol,
        "Sharpe": sharpe, "MaxDD": maxdd, "Calmar": calmar,
        "Trades": trades, "equity": eq, "equity_mkt": eq_mkt,
    }


def main() -> None:
    print("Fetching SPY data ...")
    df = fetch_fmp_daily("SPY", start="2000-01-01")
    print(f"  {len(df)} bars")

    print("Computing indicators ...")
    df_p = compute_data(df)

    sig = df_p["signal"]
    rsq_gate = (df_p["rsquare"] > 0.9).fillna(False)
    always_true = pd.Series(True, index=df_p.index)
    v = df_p["volume"]
    c = df_p["close"]

    # ===================================================================
    # Single filters
    # ===================================================================
    vol_filters: list[tuple[str, pd.Series]] = [
        ("Vol > SMA(vol,5)", v > df_p["vol_sma_5"]),
        ("Vol > SMA(vol,10)", v > df_p["vol_sma_10"]),
        ("Vol > SMA(vol,20)", v > df_p["vol_sma_20"]),
        ("Vol > SMA(vol,50)", v > df_p["vol_sma_50"]),
        ("Vol > EMA(vol,10)", v > df_p["vol_ema_10"]),
        ("Vol > EMA(vol,20)", v > df_p["vol_ema_20"]),
        ("Vol z-score > 0", df_p["vol_zscore"] > 0),
        ("Vol z-score > 0.5", df_p["vol_zscore"] > 0.5),
        ("Vol ratio > 1.2", df_p["vol_ratio_20"] > 1.2),
        ("Vol ratio > 1.5", df_p["vol_ratio_20"] > 1.5),
        ("OBV slope > 0 (20d)", df_p["obv_slope_20"] > 0),
        ("VWAP(20) < Close", df_p["vwap_20"] < c),
    ]

    ma_filters: list[tuple[str, pd.Series]] = [
        ("Close > SMA(10)", c > df_p["sma_10"]),
        ("Close > SMA(20)", c > df_p["sma_20"]),
        ("Close > SMA(50)", c > df_p["sma_50"]),
        ("Close > EMA(10)", c > df_p["ema_10"]),
        ("Close > EMA(20)", c > df_p["ema_20"]),
        ("Close > EMA(50)", c > df_p["ema_50"]),
        ("SMA(10) > SMA(50)", df_p["sma_10"] > df_p["sma_50"]),
        ("SMA(20) > SMA(50)", df_p["sma_20"] > df_p["sma_50"]),
        ("EMA(12) > EMA(26)", df_p["ema_12"] > df_p["ema_26"]),
        ("SMA(20) slope > 0", df_p["sma20_slope"] > 0),
        ("SMA(50) slope > 0", df_p["sma50_slope"] > 0),
        ("MACD hist > 0", df_p["macd_hist"] > 0),
    ]

    all_single = [("No filter (baseline)", always_true)] + vol_filters + ma_filters

    print(f"\n{'='*100}")
    print(f"Testing {len(all_single)} single filters on β_z×R²×β + R²>0.9 + 0.7/-0.8")
    print(f"{'='*100}")

    single_results = []
    for label, gate in all_single:
        gate = gate.fillna(False)
        res = backtest(sig, rsq_gate, gate, df_p["daily_ret"])
        if res:
            res["label"] = label
            single_results.append(res)
            cat = "VOL" if label.startswith(("Vol", "OBV", "VWAP")) else ("MA" if label != "No filter (baseline)" else "BASE")
            print(f"  [{cat:4s}] {label:30s} Sharpe={res['Sharpe']:.3f}  Total={res['Total']:.1%}  MaxDD={res['MaxDD']:.2%}  Calmar={res['Calmar']:.3f}  Trades={res['Trades']}")

    # ===================================================================
    # Combination filters: top volume × top MA
    # ===================================================================
    # Select top 5 volume and top 5 MA by Sharpe
    vol_results = [(r["label"], r["Sharpe"]) for r in single_results if r["label"] in [f[0] for f in vol_filters]]
    ma_results = [(r["label"], r["Sharpe"]) for r in single_results if r["label"] in [f[0] for f in ma_filters]]
    vol_results.sort(key=lambda x: x[1], reverse=True)
    ma_results.sort(key=lambda x: x[1], reverse=True)

    top_vol = vol_results[:5]
    top_ma = ma_results[:5]

    vol_dict = {name: gate.fillna(False) for name, gate in vol_filters}
    ma_dict = {name: gate.fillna(False) for name, gate in ma_filters}

    print(f"\n{'='*100}")
    print(f"Testing {len(top_vol)} × {len(top_ma)} = {len(top_vol) * len(top_ma)} volume × MA combos")
    print(f"{'='*100}")

    combo_results = []
    for vol_name, _ in top_vol:
        for ma_name, _ in top_ma:
            combo_gate = vol_dict[vol_name] & ma_dict[ma_name]
            label = f"{vol_name} & {ma_name}"
            res = backtest(sig, rsq_gate, combo_gate, df_p["daily_ret"])
            if res:
                res["label"] = label
                combo_results.append(res)
                print(f"  {label:55s} Sharpe={res['Sharpe']:.3f}  Total={res['Total']:.1%}  MaxDD={res['MaxDD']:.2%}  Calmar={res['Calmar']:.3f}")

    all_results = single_results + combo_results
    all_results.sort(key=lambda r: r.get("Sharpe", 0), reverse=True)

    # --- Summary table ---
    rows = []
    for r in all_results:
        rows.append({
            "Filter": r["label"],
            "Total": f"{r['Total']:.1%}",
            "Ann ret": f"{r['Ann ret']:.2%}",
            "Ann vol": f"{r['Ann vol']:.2%}",
            "Sharpe": f"{r['Sharpe']:.3f}",
            "MaxDD": f"{r['MaxDD']:.2%}",
            "Calmar": f"{r['Calmar']:.3f}",
            "Trades": r["Trades"],
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(RESULTS_DIR / "volume_ma_confirm.csv", index=False)
    print(f"\nFull table ({len(rows)} entries) saved.")

    # --- Charts ---
    print("\nGenerating charts ...")

    # 1. Single filter Sharpe comparison (bar chart)
    fig1, (ax_v, ax_m) = plt.subplots(1, 2, figsize=(18, 8))

    # Volume filters
    vol_labels = [r["label"] for r in single_results if r["label"] in vol_dict]
    vol_sharpes = [r["Sharpe"] for r in single_results if r["label"] in vol_dict]
    baseline_sharpe = single_results[0]["Sharpe"]
    colors_v = ["tab:green" if s > baseline_sharpe else "tab:blue" for s in vol_sharpes]
    ax_v.barh(vol_labels, vol_sharpes, color=colors_v, alpha=0.8)
    ax_v.axvline(baseline_sharpe, color="red", ls="--", lw=1.5, label=f"Baseline ({baseline_sharpe:.3f})")
    ax_v.set_xlabel("Sharpe Ratio")
    ax_v.set_title("Volume Confirmation Filters", fontsize=12)
    ax_v.legend(fontsize=9)

    # MA filters
    ma_labels = [r["label"] for r in single_results if r["label"] in ma_dict]
    ma_sharpes = [r["Sharpe"] for r in single_results if r["label"] in ma_dict]
    colors_m = ["tab:green" if s > baseline_sharpe else "tab:blue" for s in ma_sharpes]
    ax_m.barh(ma_labels, ma_sharpes, color=colors_m, alpha=0.8)
    ax_m.axvline(baseline_sharpe, color="red", ls="--", lw=1.5, label=f"Baseline ({baseline_sharpe:.3f})")
    ax_m.set_xlabel("Sharpe Ratio")
    ax_m.set_title("MA Confirmation Filters", fontsize=12)
    ax_m.legend(fontsize=9)

    plt.suptitle("β_z×R²×β + R²>0.9 + 0.7/-0.8 — Single Filter Comparison", fontsize=14)
    plt.tight_layout()
    fig1.savefig(RESULTS_DIR / "vol_ma_single_sharpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2. Top 10 equity curves
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    top10 = all_results[:10]
    for r in top10:
        r["equity"].plot(ax=ax2, label=f"{r['label']} (S={r['Sharpe']:.3f})", linewidth=1.1, alpha=0.8)
    single_results[0]["equity"].plot(ax=ax2, label=f"Baseline (S={baseline_sharpe:.3f})", color="black", ls="--", lw=2)
    single_results[0]["equity_mkt"].plot(ax=ax2, label="SPY Buy & Hold", color="gray", ls=":", lw=1.5)
    ax2.set_title("Top 10 Volume/MA Confirmations — Equity Curves", fontsize=14)
    ax2.set_ylabel("Growth of $1")
    ax2.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "vol_ma_top10_equity.png", dpi=150)
    plt.close(fig2)

    # 3. Combo heatmap (top vol × top ma by Sharpe)
    combo_sharpes = {}
    for r in combo_results:
        combo_sharpes[r["label"]] = r["Sharpe"]
    heat_data = np.full((len(top_ma), len(top_vol)), np.nan)
    for i, (ma_n, _) in enumerate(top_ma):
        for j, (vol_n, _) in enumerate(top_vol):
            key = f"{vol_n} & {ma_n}"
            if key in combo_sharpes:
                heat_data[i, j] = combo_sharpes[key]
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=[n for n, _ in top_vol],
                yticklabels=[n for n, _ in top_ma], ax=ax3)
    ax3.set_title("Sharpe: Top Volume × Top MA Combinations", fontsize=13)
    ax3.set_xlabel("Volume Filter")
    ax3.set_ylabel("MA Filter")
    plt.tight_layout()
    fig3.savefig(RESULTS_DIR / "vol_ma_combo_heatmap.png", dpi=150)
    plt.close(fig3)

    # 4. Drawdown comparison (top 5 + baseline)
    fig4, ax4 = plt.subplots(figsize=(15, 5))
    for r in all_results[:5]:
        dd = (r["equity"] / r["equity"].cummax() - 1) * 100
        dd.plot(ax=ax4, label=r["label"], alpha=0.7, lw=1)
    dd_base = (single_results[0]["equity"] / single_results[0]["equity"].cummax() - 1) * 100
    dd_base.plot(ax=ax4, label="Baseline", color="black", ls="--", lw=1.5)
    ax4.set_title("Drawdown: Top 5 Filters vs Baseline", fontsize=13)
    ax4.set_ylabel("Drawdown (%)")
    ax4.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    fig4.savefig(RESULTS_DIR / "vol_ma_drawdown.png", dpi=150)
    plt.close(fig4)

    print(f"All charts saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
