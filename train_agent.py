"""
Trains the DDPG agent (Chapter 3) on historical data and saves the actor's
weights, then exposes get_daily_returns() for backtest_harness.py's
agent_returns() to call directly.

Trains on 2015-2018 by default -- deliberately BEFORE both backtest windows
(2019, 2020) used in backtest_harness.py, so the evaluation stays
out-of-sample rather than testing the agent on data it trained on.

Usage (run once for each arm of the Chapter 5 ablation):
    python3 train_agent.py --sentiment      # trains the with-sentiment agent
    python3 train_agent.py --no-sentiment   # trains the without-sentiment agent
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch

from portfolio_env import PortfolioEnv
from ddpg_agent import DDPGAgent

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # must match backtest_harness.py


def train(start, end, use_sentiment, sentiment_series=None, episodes=20,
          save_path=None, seed=0, transaction_cost=0.0):
    # NOTE: seeding all three RNGs matters -- numpy alone (as this used to
    # do) leaves the actor/critic weight initialisation (torch) and replay
    # buffer sampling (python's random) uncontrolled, so "the same seed"
    # wasn't actually reproducible before this fix. See Chapter 5.7.
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    env = PortfolioEnv(TICKERS, start, end, use_sentiment=use_sentiment,
                        sentiment_series=sentiment_series, transaction_cost=transaction_cost)
    agent = DDPGAgent(env.state_dim, env.action_dim)

    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            noise_scale = max(0.02, 0.3 * (1 - ep / episodes))  # decay exploration
            action = agent.act(state, noise_scale=noise_scale)
            next_state, reward, done, info = env.step(action)
            s2_store = next_state if next_state is not None else state
            agent.buffer.push(state, action, reward, s2_store, float(done))
            agent.train_step()
            state = s2_store
            ep_reward += reward
        print(f"episode {ep + 1}/{episodes}  sum of daily returns = {ep_reward:.4f}")

    if save_path:
        agent.save(save_path)
        print(f"saved actor weights to {save_path}")
    return agent


def get_daily_returns(start, end, use_sentiment, weights_path, sentiment_series=None, transaction_cost=0.0):
    """
    Used by backtest_harness.py: runs the trained (frozen) actor once,
    greedily (no exploration noise), over [start, end]. Returns a pandas
    Series of daily portfolio returns -- the same format as
    equal_weight_returns() / benchmark_returns() -- so it plugs straight
    into eval_metrics.summarize().

    transaction_cost > 0 reports the NET return (after proportional cost on
    daily turnover) rather than the gross return, so backtest performance
    is not interpreted too optimistically (draft-report feedback, Chapter 5.4).
    """
    env = PortfolioEnv(TICKERS, start, end, use_sentiment=use_sentiment,
                        sentiment_series=sentiment_series, transaction_cost=transaction_cost)
    agent = DDPGAgent(env.state_dim, env.action_dim)
    agent.load(weights_path)

    state = env.reset()
    rets, dates = [], []
    done = False
    return_key = "net_return" if transaction_cost > 0 else "portfolio_return"
    while not done:
        action = agent.act(state, noise_scale=0.0)
        next_state, reward, done, info = env.step(action)
        rets.append(info[return_key])
        dates.append(info["date"])
        state = next_state if next_state is not None else state
    return pd.Series(rets, index=pd.DatetimeIndex(dates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", dest="use_sentiment", action="store_true")
    parser.add_argument("--no-sentiment", dest="use_sentiment", action="store_false")
    parser.set_defaults(use_sentiment=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2018-12-31")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0,
                         help="Run with a few different seeds (e.g. 0, 1, 2) to check "
                              "how much results vary by chance -- see Chapter 5.7.")
    args = parser.parse_args()

    suffix = "with_sentiment" if args.use_sentiment else "no_sentiment"
    if args.seed != 0:
        suffix += f"_seed{args.seed}"
    train(args.start, args.end, args.use_sentiment, episodes=args.episodes,
          save_path=f"ddpg_actor_{suffix}.pt", seed=args.seed)
