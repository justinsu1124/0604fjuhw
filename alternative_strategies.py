#!/usr/bin/env python3
"""Alternative strategy variants: Short-Selling, VIX Hedging, Bond Rotation.

Compares the base RSRS long-only strategy with:
1. Long/Short: go short when signal < sell threshold
2. Bond Rotation: hold TLT when not in SPY
3. VIX Hedging: hold VXX when signal is strongly negative

Usage
-----
    FMP_API_KEY=<key> python alternative_strategies.py
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

DOCS_DIR = Path(__file__).parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLS_WINDOW = 18
ZSCORE_WINDOW = 1100
BUY_T = 0.7
SELL_T = -0.8
TC_RATE = 5 / 10_000


def compute_rsrs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSRS signal from price DataFrame."""
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


def get_positions(signal: pd.Series, rsquare: pd.Series,
                  volume: pd.Series, vol_sma: pd.Series,
                  mode: str = "long_only") -> pd.Series:
    """Generate position series based on mode.

    Modes:
    - long_only: +1 when buy signal, 0 otherwise (original)
    - long_short: +1 when buy, -1 when sell, 0 in between
    - bond_rotation: +1 when buy (SPY), uses separate return for off periods
    - vix_hedge: +1 when buy (SPY), uses separate return for strong sell
    """
    pos = []
    prev = 0
    for s, r2, v, vsma in zip(signal, rsquare, volume, vol_sma):
        if pd.isna(s):
            pos.append(prev)
            continue
        gate = True
        if r2 <= 0.9:
            gate = False
        if pd.isna(vsma) or v <= vsma:
            gate = False

        if mode == "long_short":
            if s > BUY_T and gate:
                prev = 1
            elif s < SELL_T:
                prev = -1
            elif prev == -1 and s > 0:
                # Cover short when signal turns neutral
                prev = 0
        else:
            # long_only, bond_rotation, vix_hedge
            if s > BUY_T and gate:
                prev = 1
            elif s < SELL_T:
                prev = 0

        pos.append(prev)

    return pd.Series(pos, index=signal.index, dtype=float)


def compute_strategy_returns(pos_series: pd.Series, spy_ret: pd.Series,
                             alt_ret: pd.Series | None = None,
                             mode: str = "long_only") -> pd.Series:
    """Compute strategy returns based on position and mode."""
    position = pos_series.shift(1).fillna(0)
    delta = position.diff().abs()
    delta.iloc[0] = abs(position.iloc[0])
    tc = delta.fillna(0) * TC_RATE

    if mode == "long_only":
        strat_ret = position * spy_ret - tc
    elif mode == "long_short":
        strat_ret = position * spy_ret - tc
    elif mode == "bond_rotation":
        # When position == 1: hold SPY; when position == 0: hold TLT
        if alt_ret is not None:
            strat_ret = position * spy_ret + (1 - position.abs()) * alt_ret - tc
        else:
            strat_ret = position * spy_ret - tc
    elif mode == "vix_hedge":
        # When position == 1: hold SPY; when position == 0: hold partial VXX
        if alt_ret is not None:
            # 30% of capital in VXX during off-signal periods
            vix_alloc = 0.3
            strat_ret = (position * spy_ret +
                         (1 - position.abs()) * vix_alloc * alt_ret - tc)
        else:
            strat_ret = position * spy_ret - tc
    else:
        strat_ret = position * spy_ret - tc

    return strat_ret


def calc_metrics(eq: pd.Series) -> dict:
    """Calculate performance metrics from equity curve."""
    rets = eq.pct_change().dropna()
    n = len(rets)
    if n < 50:
        return {}
    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (252 / n) - 1
    vol = rets.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = eq / eq.cummax() - 1
    maxdd = dd.min()
    calmar = ann / abs(maxdd) if maxdd < 0 else 0
    return {"Total": total, "Ann": ann, "Vol": vol, "Sharpe": sharpe,
            "MaxDD": maxdd, "Calmar": calmar}


def main() -> None:
    print("=" * 80)
    print("Alternative Strategy Variants: Short / Bond Rotation / VIX Hedge")
    print("=" * 80)

    # Fetch data
    print("\nFetching data ...")
    spy = fetch_fmp_daily("SPY", start="2000-01-01")
    tlt = fetch_fmp_daily("TLT", start="2000-01-01")
    vxx = fetch_fmp_daily("VXX", start="2000-01-01")
    print(f"  SPY: {len(spy)} bars")
    print(f"  TLT: {len(tlt)} bars")
    print(f"  VXX: {len(vxx)} bars")

    # Compute RSRS on SPY
    print("\nComputing RSRS ...")
    spy_rsrs = compute_rsrs(spy)

    # Prepare alt returns aligned to SPY index
    tlt_p = tlt.set_index("date").sort_index()
    tlt_p["daily_ret"] = tlt_p["open"].pct_change()
    vxx_p = vxx.set_index("date").sort_index()
    vxx_p["daily_ret"] = vxx_p["open"].pct_change()

    # Align to SPY index
    tlt_ret_aligned = tlt_p["daily_ret"].reindex(spy_rsrs.index).fillna(0)
    vxx_ret_aligned = vxx_p["daily_ret"].reindex(spy_rsrs.index).fillna(0)

    # Get signal start
    first_valid = spy_rsrs["signal"].first_valid_index()

    # --- Strategy 1: Long-Only (baseline) ---
    print("\nRunning strategies ...")
    pos_long = get_positions(spy_rsrs["signal"], spy_rsrs["rsquare"],
                             spy_rsrs["volume"], spy_rsrs["vol_sma_50"],
                             mode="long_only")
    ret_long = compute_strategy_returns(pos_long, spy_rsrs["daily_ret"],
                                        mode="long_only")

    # --- Strategy 2: Long/Short ---
    pos_ls = get_positions(spy_rsrs["signal"], spy_rsrs["rsquare"],
                           spy_rsrs["volume"], spy_rsrs["vol_sma_50"],
                           mode="long_short")
    ret_ls = compute_strategy_returns(pos_ls, spy_rsrs["daily_ret"],
                                      mode="long_short")

    # --- Strategy 3: Bond Rotation (Long SPY or TLT) ---
    pos_bond = get_positions(spy_rsrs["signal"], spy_rsrs["rsquare"],
                             spy_rsrs["volume"], spy_rsrs["vol_sma_50"],
                             mode="bond_rotation")
    ret_bond = compute_strategy_returns(pos_bond, spy_rsrs["daily_ret"],
                                        alt_ret=tlt_ret_aligned,
                                        mode="bond_rotation")

    # --- Strategy 4: VIX Hedge (Long SPY + partial VXX in off periods) ---
    pos_vix = get_positions(spy_rsrs["signal"], spy_rsrs["rsquare"],
                            spy_rsrs["volume"], spy_rsrs["vol_sma_50"],
                            mode="vix_hedge")
    ret_vix = compute_strategy_returns(pos_vix, spy_rsrs["daily_ret"],
                                       alt_ret=vxx_ret_aligned,
                                       mode="vix_hedge")

    # Compute equity curves from first valid signal
    strategies = {
        "Long-Only (Baseline)": ret_long,
        "Long/Short": ret_ls,
        "Bond Rotation (SPY+TLT)": ret_bond,
        "VIX Hedge (SPY+30%VXX)": ret_vix,
    }

    results = {}
    equities = {}
    for name, ret in strategies.items():
        r = ret.loc[first_valid:]
        eq = (1 + r.fillna(0)).cumprod()
        equities[name] = eq
        metrics = calc_metrics(eq)
        results[name] = metrics
        if metrics:
            print(f"  {name:<30}: Sharpe={metrics['Sharpe']:.3f} "
                  f"Total={metrics['Total']:.1%} MaxDD={metrics['MaxDD']:.2%} "
                  f"Calmar={metrics['Calmar']:.3f}")

    # Buy & Hold
    mkt_ret = spy_rsrs["daily_ret"].loc[first_valid:]
    eq_mkt = (1 + mkt_ret.fillna(0)).cumprod()
    equities["Buy & Hold"] = eq_mkt
    results["Buy & Hold"] = calc_metrics(eq_mkt)

    # --- Plot comparison ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[3, 1])

    colors = {
        "Long-Only (Baseline)": "tab:blue",
        "Long/Short": "tab:red",
        "Bond Rotation (SPY+TLT)": "tab:green",
        "VIX Hedge (SPY+30%VXX)": "tab:purple",
        "Buy & Hold": "black",
    }
    styles = {
        "Long-Only (Baseline)": "-",
        "Long/Short": "-",
        "Bond Rotation (SPY+TLT)": "-",
        "VIX Hedge (SPY+30%VXX)": "-",
        "Buy & Hold": "--",
    }

    for name, eq in equities.items():
        m = results.get(name, {})
        label = f"{name} (S={m.get('Sharpe', 0):.3f}, DD={m.get('MaxDD', 0):.1%})"
        ax1.plot(eq.index, eq.values, label=label,
                 color=colors.get(name, "gray"),
                 ls=styles.get(name, "-"),
                 lw=2 if name != "Buy & Hold" else 1.5,
                 alpha=0.9 if name != "Buy & Hold" else 0.7)

    ax1.set_title("RSRS Strategy Variants — Equity Curve Comparison (SPY 2004-2026)",
                  fontsize=14, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Drawdown comparison
    for name, eq in equities.items():
        if name == "Buy & Hold":
            continue
        dd = eq / eq.cummax() - 1
        ax2.plot(dd.index, dd.values, label=name,
                 color=colors.get(name, "gray"), lw=1.2, alpha=0.8)

    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_title("Drawdown Comparison", fontsize=12)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(DOCS_DIR / "strategy_variants.png", dpi=150)
    plt.close(fig)
    print("\n  Saved strategy_variants.png")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<32} {'Sharpe':>7} {'Total':>8} {'MaxDD':>8} {'Calmar':>7}")
    print("-" * 70)
    for name in ["Long-Only (Baseline)", "Long/Short",
                 "Bond Rotation (SPY+TLT)", "VIX Hedge (SPY+30%VXX)",
                 "Buy & Hold"]:
        m = results.get(name, {})
        if m:
            print(f"  {name:<30} {m['Sharpe']:>7.3f} {m['Total']:>7.1%} "
                  f"{m['MaxDD']:>7.2%} {m['Calmar']:>7.3f}")

    # Save CSV
    rows = []
    for name, m in results.items():
        if m:
            rows.append({
                "Strategy": name,
                "Sharpe": f"{m['Sharpe']:.3f}",
                "Total": f"{m['Total']:.1%}",
                "MaxDD": f"{m['MaxDD']:.2%}",
                "Calmar": f"{m['Calmar']:.3f}",
                "Ann_Ret": f"{m['Ann']:.2%}",
                "Ann_Vol": f"{m['Vol']:.2%}",
            })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "strategy_variants.csv", index=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
