"""
Trains both arms (with/without sentiment) across several fixed seeds, so the
sentiment ablation (Chapter 5.7) can eventually be judged on a distribution
of outcomes rather than one training run per arm.

This directly follows the draft-report feedback: "run the two arms across
multiple fully controlled seeds ... report distributions rather than one
training run."

Each (arm, seed) pair gets its own weights file:
    ddpg_actor_with_sentiment_seed0.pt, ddpg_actor_with_sentiment_seed1.pt, ...
    ddpg_actor_no_sentiment_seed0.pt,   ddpg_actor_no_sentiment_seed1.pt, ...

Usage:
    python3 train_multi_seed.py --seeds 0 1 2 3 4
    python3 train_multi_seed.py --seeds 0 1 2 3 4 --transaction-cost 0.0005

Note: this runs 2 arms x N seeds = 2N full training runs. With the default
5 seeds that's 10 runs -- expect this to take roughly 10x as long as a single
`train_agent.py` run. Start with fewer seeds (e.g. --seeds 0 1 2) if that's
too slow, and add more later; each seed's weights file is independent, so
extending the seed list later doesn't invalidate what's already been trained.
"""
import argparse
import os
import time

import pandas as pd

from train_agent import train, TICKERS

SENTIMENT_CSV = "sentiment_series.csv"


def load_sentiment_series(path=SENTIMENT_CSV):
    """
    Loads the output of load_news_data.py if present. Returns None if the
    file doesn't exist yet, so every script here still runs (with sentiment
    defaulting to neutral/zero, Chapter 3.3) before real news data is wired in.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    series = df.iloc[:, 0]
    print(f"Loaded real sentiment series from {path}: {len(series)} dates "
          f"({series.index.min().date()} to {series.index.max().date()}), "
          f"mean={series.mean():.4f}.")
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--transaction-cost", type=float, default=0.0,
                         help="Proportional cost per unit of daily turnover, e.g. 0.0005 = 5bps.")
    args = parser.parse_args()

    total_runs = 2 * len(args.seeds)
    run_i = 0
    t0 = time.time()
    sentiment_series = load_sentiment_series()
    if sentiment_series is None:
        print(f"No {SENTIMENT_CSV} found -- training the 'with sentiment' arm with an "
              f"all-zero sentiment feature (Chapter 3.3's known limitation). "
              f"Run load_news_data.py first for a real ablation.")

    for use_sentiment in [True, False]:
        arm = "with_sentiment" if use_sentiment else "no_sentiment"
        for seed in args.seeds:
            run_i += 1
            save_path = f"ddpg_actor_{arm}_seed{seed}.pt"
            elapsed = time.time() - t0
            print(f"\n=== [{run_i}/{total_runs}] arm={arm} seed={seed} "
                  f"-> {save_path}  (elapsed so far: {elapsed/60:.1f} min) ===")
            train(args.start, args.end, use_sentiment, episodes=args.episodes,
                  save_path=save_path, seed=seed, transaction_cost=args.transaction_cost,
                  sentiment_series=sentiment_series if use_sentiment else None)

    print(f"\nAll {total_runs} runs complete in {(time.time() - t0) / 60:.1f} minutes.")
    print("Next: python3 evaluate_multi_seed.py --seeds " + " ".join(str(s) for s in args.seeds))


if __name__ == "__main__":
    main()
