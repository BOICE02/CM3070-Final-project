"""
Evaluates every trained seed (train_multi_seed.py) across both backtest
periods, and compares the with/without-sentiment arms as a DISTRIBUTION
across seeds rather than a single point estimate.

This directly follows the draft-report feedback: "use confidence intervals
or an appropriate statistical comparison to show whether any effect is
larger than ordinary DDPG variance."

Usage:
    python3 evaluate_multi_seed.py --seeds 0 1 2 3 4
    python3 evaluate_multi_seed.py --seeds 0 1 2 3 4 --transaction-cost 0.0005

Output:
    multi_seed_results.csv   -- one row per (period, arm, seed), all 5 metrics
    printed summary          -- mean +/- 95% CI per arm/period, and a paired
                                 comparison (with vs without sentiment) per period
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from train_agent import get_daily_returns
from train_multi_seed import load_sentiment_series
from backtest_harness import equal_weight_returns, benchmark_returns, get_price_data, TICKERS, BENCHMARK
from eval_metrics import summarize

PERIODS = {
    "2020_volatile": ("2020-01-01", "2020-12-31"),
    "2019_calm": ("2019-01-01", "2019-12-31"),
}


def collect_agent_results(seeds, transaction_cost=0.0):
    sentiment_series = load_sentiment_series()
    rows = []
    for period_label, (start, end) in PERIODS.items():
        for use_sentiment in [True, False]:
            arm = "with_sentiment" if use_sentiment else "no_sentiment"
            for seed in seeds:
                weights_path = f"ddpg_actor_{arm}_seed{seed}.pt"
                try:
                    returns = get_daily_returns(
                        start, end, use_sentiment, weights_path,
                        transaction_cost=transaction_cost,
                        sentiment_series=sentiment_series if use_sentiment else None,
                    )
                except FileNotFoundError:
                    print(f"  [skip] {weights_path} not found -- run train_multi_seed.py first")
                    continue
                metrics = summarize(returns, f"{arm}_seed{seed}")
                metrics.update(period=period_label, arm=arm, seed=seed)
                rows.append(metrics)
    return pd.DataFrame(rows)


def collect_baseline_results(transaction_cost=0.0):
    rows = []
    for period_label, (start, end) in PERIODS.items():
        prices = get_price_data(TICKERS, start, end)
        ew = equal_weight_returns(prices, transaction_cost=transaction_cost)
        bench = benchmark_returns(BENCHMARK, start, end)
        for label, returns in [("equal_weight", ew), ("sp500", bench)]:
            metrics = summarize(returns, label)
            metrics.update(period=period_label, arm=label, seed=None)
            rows.append(metrics)
    return pd.DataFrame(rows)


def mean_ci(vals, confidence=0.95):
    vals = np.asarray(vals, dtype=float)
    mean = vals.mean()
    if len(vals) > 1 and vals.std(ddof=1) > 0:
        sem = stats.sem(vals)
        lo, hi = stats.t.interval(confidence, len(vals) - 1, loc=mean, scale=sem)
    else:
        lo = hi = mean
    return mean, lo, hi


def report_distribution(df, period, metric="sharpe_ratio"):
    print(f"\n--- {period}: {metric} across seeds ---")
    for arm in ["with_sentiment", "no_sentiment"]:
        vals = df[(df.period == period) & (df.arm == arm)].sort_values("seed")[metric].values
        if len(vals) == 0:
            print(f"  {arm}: no trained seeds found yet")
            continue
        mean, lo, hi = mean_ci(vals)
        print(f"  {arm:15s} n={len(vals)}  mean={mean:7.3f}  95% CI=({lo:7.3f}, {hi:7.3f})  "
              f"values={np.round(vals, 3).tolist()}")


def compare_arms(df, period, metric="sharpe_ratio"):
    with_df = df[(df.period == period) & (df.arm == "with_sentiment")].sort_values("seed")
    without_df = df[(df.period == period) & (df.arm == "no_sentiment")].sort_values("seed")
    # Pair by seed so the comparison is like-for-like (same seed, same training data).
    paired = with_df.merge(without_df, on="seed", suffixes=("_with", "_without"))
    n = len(paired)
    print(f"\n--- {period}: with vs. without sentiment, paired by seed ---")
    if n < 2:
        print(f"  Only {n} paired seed(s) available -- need at least 2 for a paired test. "
              f"Train more seeds with train_multi_seed.py.")
        return
    with_vals = paired[f"{metric}_with"].values
    without_vals = paired[f"{metric}_without"].values
    diff = with_vals - without_vals
    t_stat, p_value = stats.ttest_rel(with_vals, without_vals)
    mean, lo, hi = mean_ci(diff)
    print(f"  n={n} paired seeds. Mean difference (with - without) = {mean:.4f}, "
          f"95% CI=({lo:.4f}, {hi:.4f})")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if n < 8:
        print(f"  NOTE: n={n} seeds gives this test low statistical power -- treat the "
              f"p-value as indicative, not conclusive, until more seeds are run.")
    if lo <= 0 <= hi:
        print(f"  The 95% CI includes zero: no significant difference detected at this sample size.")
    else:
        print(f"  The 95% CI excludes zero: a significant difference at the 95% level.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--transaction-cost", type=float, default=0.0,
                         help="Must match what train_multi_seed.py was run with, if nonzero.")
    args = parser.parse_args()

    agent_df = collect_agent_results(args.seeds, transaction_cost=args.transaction_cost)
    if agent_df.empty:
        print("No trained seeds found. Run train_multi_seed.py first.")
        return

    baseline_df = collect_baseline_results(transaction_cost=args.transaction_cost)
    full_df = pd.concat([agent_df, baseline_df], ignore_index=True)
    full_df.to_csv("multi_seed_results.csv", index=False)
    print(f"Saved {len(full_df)} rows to multi_seed_results.csv "
          f"({len(agent_df)} agent runs, {len(baseline_df)} baseline rows)")

    for period in PERIODS:
        print(f"\n{'=' * 60}\n{period}\n{'=' * 60}")
        ew = baseline_df[(baseline_df.period == period) & (baseline_df.arm == "equal_weight")]
        sp = baseline_df[(baseline_df.period == period) & (baseline_df.arm == "sp500")]
        if not ew.empty:
            print(f"  equal_weight    sharpe={ew.sharpe_ratio.iloc[0]:.3f}  "
                  f"cum_return={ew.cumulative_return.iloc[0]:.3f}")
        if not sp.empty:
            print(f"  sp500           sharpe={sp.sharpe_ratio.iloc[0]:.3f}  "
                  f"cum_return={sp.cumulative_return.iloc[0]:.3f}")
        report_distribution(agent_df, period, "sharpe_ratio")
        report_distribution(agent_df, period, "cumulative_return")
        compare_arms(agent_df, period, "sharpe_ratio")


if __name__ == "__main__":
    main()
