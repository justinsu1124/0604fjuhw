#!/usr/bin/env python3
"""Combinatorially Symmetric Cross-Validation (CSCV) for overfitting detection.

Implements the PBO (Probability of Backtest Overfitting) framework from
Marcos López de Prado (2018). Tests whether the best-performing strategy
configuration was selected by luck (overfitting) or genuine skill.

Method:
  1. Partition the time series into S non-overlapping blocks.
  2. Enumerate all C(S, S/2) combinations of blocks as in-sample (IS) / out-of-sample (OOS).
  3. For each split, rank all N strategy configs by IS Sharpe; record the OOS Sharpe
     of the IS-best config.
  4. Compute λ = fraction of splits where the IS-best config has OOS Sharpe < median OOS Sharpe.
  5. PBO = λ. High PBO (>0.5) → likely overfitting.

Also computes the "logit" distribution: logit(rank_OOS / N) for the IS-best config.
If the distribution is centered at or above 0, the IS selection has no OOS edge → overfit.

Usage
-----
    FMP_API_KEY=<key> python cscv_test.py
"""

from __future__ import annotations

from itertools import combinations
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
# Data & indicator computation (same as combo_search)
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

    tr = pd.concat([
        df_p["high"] - df_p["low"],
        (df_p["high"] - df_p["close"].shift(1)).abs(),
        (df_p["low"] - df_p["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df_p["atr_20"] = tr.rolling(20).mean()
    df_p["atr_20_median"] = df_p["atr_20"].rolling(252).median()
    df_p["sma_50"] = df_p["close"].rolling(50).mean()
    df_p["sma_200"] = df_p["close"].rolling(200).mean()
    df_p["sma50_slope"] = df_p["sma_50"] - df_p["sma_50"].shift(10)
    df_p["vol_sma_50"] = df_p["volume"].rolling(50).mean()
    return df_p


# ---------------------------------------------------------------------------
# Strategy return matrix: one column per config, one row per day
# ---------------------------------------------------------------------------

def build_strategy_returns(df_p: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a matrix of daily strategy returns for all configs.

    Returns (ret_matrix, config_names).
    """
    always_true = pd.Series(True, index=df_p.index)

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

    first_valid = df_p["sig_right_skew"].first_valid_index()
    daily_ret = df_p["daily_ret"]

    config_names = []
    all_strat_rets = {}

    for sig_name, sig_series in signals:
        for filt_name, filt_gate in filters:
            gate = filt_gate.fillna(False)
            for thresh_name, buy_t, sell_t in thresholds:
                name = f"{sig_name}|{filt_name}|{thresh_name}"
                config_names.append(name)

                pos = []
                prev = 0
                for s, g in zip(sig_series, gate):
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
                pos_series = pd.Series(pos, index=sig_series.index, dtype=float)
                position = pos_series.shift(1).fillna(0)
                delta = position.diff().abs()
                delta.iloc[0] = abs(position.iloc[0])
                tc = delta.fillna(0) * TC_RATE
                strat_ret = position * daily_ret - tc

                all_strat_rets[name] = strat_ret

    ret_matrix = pd.DataFrame(all_strat_rets)

    # Trim to valid signal period
    if first_valid is not None:
        ret_matrix = ret_matrix.loc[first_valid:]

    ret_matrix = ret_matrix.dropna()
    return ret_matrix, config_names


# ---------------------------------------------------------------------------
# CSCV core
# ---------------------------------------------------------------------------

def sharpe_from_returns(rets: np.ndarray) -> float:
    """Annualized Sharpe from daily returns."""
    if len(rets) < 2:
        return 0.0
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    if sigma < 1e-12:
        return 0.0
    return mu / sigma * np.sqrt(252)


def run_cscv(ret_matrix: np.ndarray, n_blocks: int = 16, max_combos: int = 5000) -> dict:
    """Run CSCV on a (T, N) return matrix.

    Args:
        ret_matrix: shape (T, N) — daily returns for N configs over T days.
        n_blocks: S — number of blocks to partition into (must be even).
        max_combos: cap on number of combinations to evaluate (sample if needed).

    Returns:
        dict with PBO, logit distribution, and IS vs OOS Sharpe ranks.
    """
    T, N = ret_matrix.shape
    S = n_blocks
    assert S % 2 == 0, "n_blocks must be even"
    half = S // 2

    block_size = T // S
    blocks = []
    for i in range(S):
        start = i * block_size
        end = start + block_size if i < S - 1 else T
        blocks.append(ret_matrix[start:end])

    # All C(S, S/2) combinations
    all_combos = list(combinations(range(S), half))
    total_combos = len(all_combos)
    print(f"  Total C({S},{half}) = {total_combos} combinations")

    if total_combos > max_combos:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_combos, max_combos, replace=False)
        selected_combos = [all_combos[i] for i in sorted(indices)]
        print(f"  Sampling {max_combos} combinations")
    else:
        selected_combos = all_combos

    oos_sharpes_of_is_best = []
    oos_ranks_of_is_best = []
    logit_values = []
    is_oos_pairs = []
    below_median_count = 0

    for combo in selected_combos:
        oos_idx = tuple(i for i in range(S) if i not in combo)

        # Build IS and OOS return matrices
        is_rets = np.vstack([blocks[i] for i in combo])
        oos_rets = np.vstack([blocks[i] for i in oos_idx])

        # Compute Sharpe for each config
        is_sharpes = np.array([sharpe_from_returns(is_rets[:, j]) for j in range(N)])
        oos_sharpes = np.array([sharpe_from_returns(oos_rets[:, j]) for j in range(N)])

        # IS-best config
        is_best = np.argmax(is_sharpes)
        is_best_oos_sharpe = oos_sharpes[is_best]
        oos_sharpes_of_is_best.append(is_best_oos_sharpe)

        # PBO: compare IS-best's OOS Sharpe vs median of ALL configs' OOS Sharpe in this split
        median_oos_all = np.median(oos_sharpes)
        if is_best_oos_sharpe < median_oos_all:
            below_median_count += 1

        # Rank of IS-best in OOS (1 = best)
        sorted_oos = np.sort(oos_sharpes)
        oos_rank = N - np.searchsorted(sorted_oos, is_best_oos_sharpe, side="right")
        oos_rank = max(oos_rank, 1)  # ensure 1-indexed
        oos_ranks_of_is_best.append(oos_rank)

        # Logit of relative rank
        w = oos_rank / N
        w = np.clip(w, 0.01, 0.99)  # avoid log(0)
        logit_val = np.log(w / (1 - w))
        logit_values.append(logit_val)

        is_oos_pairs.append((is_sharpes[is_best], is_best_oos_sharpe))

    oos_sharpes_arr = np.array(oos_sharpes_of_is_best)
    pbo = below_median_count / len(selected_combos)

    # Alternative PBO: fraction where OOS rank is worse than N/2
    pbo_rank = np.mean(np.array(oos_ranks_of_is_best) > N / 2)

    return {
        "PBO": pbo,
        "PBO_rank": pbo_rank,
        "n_combos": len(selected_combos),
        "n_configs": N,
        "oos_sharpes": oos_sharpes_arr,
        "logit_values": np.array(logit_values),
        "is_oos_pairs": is_oos_pairs,
        "oos_ranks": np.array(oos_ranks_of_is_best),
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

    print("Building strategy return matrix (150 configs) ...")
    ret_matrix, config_names = build_strategy_returns(df_p)
    print(f"  Matrix shape: {ret_matrix.shape} (days × configs)")

    # Run CSCV with S=16 blocks
    print("\nRunning CSCV (S=16 blocks) ...")
    result = run_cscv(ret_matrix.values, n_blocks=16)

    pbo = result["PBO"]
    pbo_rank = result["PBO_rank"]
    print(f"\n{'='*60}")
    print("  PBO (Probability of Backtest Overfitting)")
    print(f"{'='*60}")
    print(f"  PBO (median OOS Sharpe):  {pbo:.3f}")
    print(f"  PBO (rank-based):         {pbo_rank:.3f}")
    print(f"  Combos tested:            {result['n_combos']}")
    print(f"  Strategy configs:         {result['n_configs']}")
    print(f"  Mean OOS Sharpe of IS-best: {result['oos_sharpes'].mean():.3f}")
    print(f"  Median OOS Sharpe of IS-best: {np.median(result['oos_sharpes']):.3f}")
    print(f"  Mean logit:               {result['logit_values'].mean():.3f}")
    print(f"{'='*60}")

    if pbo < 0.25:
        verdict = "LOW overfitting risk — strategy selection likely reflects genuine edge."
    elif pbo < 0.50:
        verdict = "MODERATE overfitting risk — some edge likely, but selection bias present."
    else:
        verdict = "HIGH overfitting risk — IS-best config may not outperform OOS."
    print(f"\n  Verdict: {verdict}\n")

    # --- Charts ---
    print("Generating CSCV charts ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Logit distribution
    ax = axes[0, 0]
    ax.hist(result["logit_values"], bins=40, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=2, label="logit=0 (random)")
    ax.axvline(result["logit_values"].mean(), color="green", ls="-", lw=2,
               label=f"mean={result['logit_values'].mean():.2f}")
    ax.set_title("CSCV Logit Distribution", fontsize=13)
    ax.set_xlabel("logit(rank_OOS / N)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.text(0.05, 0.95, f"PBO = {pbo:.3f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 2. OOS Sharpe distribution of IS-best
    ax = axes[0, 1]
    ax.hist(result["oos_sharpes"], bins=40, color="tab:orange", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    ax.axvline(result["oos_sharpes"].mean(), color="green", ls="-", lw=2,
               label=f"mean={result['oos_sharpes'].mean():.2f}")
    ax.set_title("OOS Sharpe of IS-Best Config", fontsize=13)
    ax.set_xlabel("OOS Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.legend()

    # 3. IS Sharpe vs OOS Sharpe scatter
    ax = axes[1, 0]
    is_vals = [p[0] for p in result["is_oos_pairs"]]
    oos_vals = [p[1] for p in result["is_oos_pairs"]]
    ax.scatter(is_vals, oos_vals, alpha=0.3, s=15, color="tab:purple")
    mn = min(min(is_vals), min(oos_vals))
    mx = max(max(is_vals), max(oos_vals))
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="IS = OOS")
    ax.set_title("IS Sharpe vs OOS Sharpe (IS-best config)", fontsize=13)
    ax.set_xlabel("In-Sample Sharpe")
    ax.set_ylabel("Out-of-Sample Sharpe")
    ax.legend()

    # 4. OOS rank distribution
    ax = axes[1, 1]
    ax.hist(result["oos_ranks"], bins=range(1, result["n_configs"] + 2),
            color="tab:green", alpha=0.7, edgecolor="white")
    ax.axvline(result["n_configs"] / 2, color="red", ls="--", lw=1.5, label="Median rank")
    ax.set_title("OOS Rank of IS-Best Config", fontsize=13)
    ax.set_xlabel(f"Rank (1=best, {result['n_configs']}=worst)")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.suptitle(f"CSCV Overfitting Analysis — PBO = {pbo:.3f}", fontsize=15)
    plt.tight_layout()
    chart_path = RESULTS_DIR / "cscv_analysis.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to {chart_path}")

    # --- Also run with different block sizes for robustness ---
    print("\nRobustness check: varying block count ...")
    block_sizes = [8, 10, 12, 16, 20]
    robustness = []
    for s in block_sizes:
        r = run_cscv(ret_matrix.values, n_blocks=s)
        robustness.append({
            "Blocks": s,
            "PBO": f"{r['PBO']:.3f}",
            "PBO_rank": f"{r['PBO_rank']:.3f}",
            "Mean OOS Sharpe": f"{r['oos_sharpes'].mean():.3f}",
            "Mean logit": f"{r['logit_values'].mean():.3f}",
        })
        print(f"  S={s:2d}: PBO={r['PBO']:.3f}, mean_logit={r['logit_values'].mean():.3f}")

    rob_df = pd.DataFrame(robustness)
    print(f"\n{rob_df.to_string(index=False)}")
    rob_df.to_csv(RESULTS_DIR / "cscv_robustness.csv", index=False)

    # Robustness chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    pbo_vals = [float(r["PBO"]) for r in robustness]
    ax2.bar([str(s) for s in block_sizes], pbo_vals, color=["green" if p < 0.25 else "orange" if p < 0.5 else "red" for p in pbo_vals], alpha=0.8)
    ax2.axhline(0.5, color="red", ls="--", lw=1.5, label="PBO = 0.5 (random)")
    ax2.axhline(0.25, color="orange", ls="--", lw=1, label="PBO = 0.25 (caution)")
    ax2.set_xlabel("Number of Blocks (S)")
    ax2.set_ylabel("PBO")
    ax2.set_title("CSCV Robustness: PBO across block sizes", fontsize=13)
    ax2.legend()
    ax2.set_ylim(0, 1)
    plt.tight_layout()
    fig2.savefig(RESULTS_DIR / "cscv_robustness.png", dpi=150)
    plt.close(fig2)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  PBO across block sizes: {[float(r['PBO']) for r in robustness]}")
    mean_pbo = np.mean([float(r["PBO"]) for r in robustness])
    print(f"  Average PBO: {mean_pbo:.3f}")
    if mean_pbo < 0.25:
        print("  → LOW overfitting risk across all block sizes.")
    elif mean_pbo < 0.50:
        print("  → MODERATE overfitting risk. Some configurations may be overfit.")
    else:
        print("  → HIGH overfitting risk. Strategy selection likely driven by noise.")
    print(f"{'='*60}")

    print("\nDone!")


if __name__ == "__main__":
    main()
