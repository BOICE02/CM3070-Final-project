"""
Backtest harness for the AI Financial Advisor Bot (CM3020 Draft Report,
Chapter 5).

Runs strategies over ONE market period (edit START/END/PERIOD_LABEL and
re-run for each period) and produces:
  - results_<period>.csv      -> one row per strategy, feeds the Ch.5 table
  - equity_curve_<period>.png -> screenshot this for Ch.4/Ch.5 figures

Strategies:
  1. Equal-weight (EW), rebalanced monthly           <- runs out of the box
  2. Buy-and-hold benchmark index (default: S&P 500)  <- runs out of the box
  3. DDPG agent WITH the sentiment feature            <- plug in your model
  4. DDPG agent WITHOUT the sentiment feature         <- plug in your model

Run once for a volatile period (e.g. 2020) and once for a calmer one
(e.g. 2018 or 2022-2023), per the plan in Chapter 3/5. That gives the two
rows per strategy the evaluation needs.

Requires: pip install yfinance pandas matplotlib
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from eval_metrics import summarize

# ---- Configuration: edit these for each run ----
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # your fixed basket (Ch.3)
BENCHMARK = "^GSPC"                                    # S&P 500
START = "2020-01-01"
END = "2020-12-31"
PERIOD_LABEL = "2020_volatile"                         # e.g. "2018_calm" next run


def get_price_data(tickers, start, end) -> pd.DataFrame:
    # threads=False avoids yfinance's occasional "database is locked" sqlite
    # error. auto_adjust=True is set explicitly (rather than relying on
    # yfinance's default, which has changed across versions) so "Close" is
    # always the dividend/split-adjusted price -- newer yfinance no longer
    # returns a separate "Adj Close" column when auto_adjust is on.
    data = yf.download(tickers, start=start, end=end, threads=False, auto_adjust=True)["Close"]
    return data.dropna()


def equal_weight_returns(prices: pd.DataFrame, rebalance: str = "ME", transaction_cost: float = 0.0) -> pd.Series:
    """Equal-weight portfolio, rebalanced at the given frequency (drifts between).
    Frequency uses pandas offset aliases: 'ME' = month-end, 'W' = weekly, etc.
    ('M' was pandas's old alias for month-end; recent pandas requires 'ME'.)
    transaction_cost > 0 charges proportional cost on rebalance-day turnover,
    matching PortfolioEnv's treatment of the agent, so the two stay comparable.
    """
    daily_returns = prices.pct_change().dropna()
    target_weights = pd.Series(1 / len(prices.columns), index=prices.columns)
    rebalance_dates = set(daily_returns.resample(rebalance).first().index)

    port_returns = []
    current_weights = target_weights.copy()
    for date, row in daily_returns.iterrows():
        cost = 0.0
        if date in rebalance_dates:
            turnover = float((target_weights - current_weights).abs().sum())
            cost = transaction_cost * turnover
            current_weights = target_weights.copy()
        port_returns.append(float((row * current_weights).sum()) - cost)
        current_weights = current_weights * (1 + row)
        current_weights /= current_weights.sum()

    return pd.Series(port_returns, index=daily_returns.index)


def benchmark_returns(benchmark_ticker: str, start: str, end: str) -> pd.Series:
    px = yf.download(benchmark_ticker, start=start, end=end, threads=False, auto_adjust=True)["Close"]
    # yfinance can return a 1-column DataFrame (multi-index columns) even for
    # a single ticker; coerce to a plain Series either way.
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.pct_change().dropna()


def agent_returns(use_sentiment: bool) -> pd.Series:
    """
    Runs the trained DDPG agent (Chapter 3) over this file's START/END
    period, using frozen weights trained separately on 2015-2018
    (train_agent.py's default) so this backtest period stays out-of-sample.

    Train the weights first (once per arm, they're reused across periods):
        python3 train_agent.py --sentiment
        python3 train_agent.py --no-sentiment
    """
    from train_agent import get_daily_returns
    suffix = "with_sentiment" if use_sentiment else "no_sentiment"
    weights_path = f"ddpg_actor_{suffix}.pt"
    return get_daily_returns(START, END, use_sentiment, weights_path)


def main():
    prices = get_price_data(TICKERS, START, END)
    results = []

    ew = equal_weight_returns(prices)
    results.append(summarize(ew, "Equal-weight"))

    bench = benchmark_returns(BENCHMARK, START, END)
    results.append(summarize(bench, "S&P 500 (buy-and-hold)"))

    try:
        agent_with = agent_returns(use_sentiment=True)
        results.append(summarize(agent_with, "DDPG + sentiment"))
        agent_without = agent_returns(use_sentiment=False)
        results.append(summarize(agent_without, "DDPG, no sentiment"))
    except FileNotFoundError:
        print("No trained agent weights found yet -- run train_agent.py first "
              "(see agent_returns() docstring). Continuing with baselines only.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results_{PERIOD_LABEL}.csv", index=False)
    print(results_df)

    plt.figure(figsize=(9, 5))
    plt.plot((1 + ew).cumprod(), label="Equal-weight")
    plt.plot((1 + bench).cumprod(), label="S&P 500")
    try:
        plt.plot((1 + agent_with).cumprod(), label="DDPG + sentiment")
        plt.plot((1 + agent_without).cumprod(), label="DDPG, no sentiment")
    except NameError:
        pass  # agent weights not trained yet -- plot baselines only
    plt.title(f"Equity curves \u2014 {PERIOD_LABEL}")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"equity_curve_{PERIOD_LABEL}.png", dpi=150)
    print(f"Saved results_{PERIOD_LABEL}.csv and equity_curve_{PERIOD_LABEL}.png")


if __name__ == "__main__":
    main()
