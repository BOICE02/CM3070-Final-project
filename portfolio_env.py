"""
Custom Gym-style environment for the AI Financial Advisor Bot's DDPG agent
(CM3020 Draft Report, Chapter 3: MDP formulation, Table 3.1).

State S_t  = [price-return history buffer | MACD, RSI | previous weights | sentiment score]
Action A_t = continuous portfolio weights on the simplex (sum to 1, no shorting)
Reward     = portfolio return for the day (no transaction costs by default,
             to match the baselines in eval_metrics.py / backtest_harness.py)
"""
import numpy as np
import pandas as pd


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return (macd / prices).fillna(0.0)  # normalise by price level


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return ((rsi.fillna(50) - 50) / 50).astype(float)  # rescale to roughly [-1, 1]


class PortfolioEnv:
    """
    Parameters
    ----------
    tickers : list[str]
    start, end : str ("YYYY-MM-DD")
    prices : pd.DataFrame, optional
        Pre-loaded adjusted-close price data (columns = tickers). If not
        given, downloaded via yfinance (threads=False, auto_adjust=True,
        matching backtest_harness.py). Passing prices in directly avoids a
        redundant download when the caller already has the data, and makes
        the environment testable offline.
    history_len : int
        Number of past days of returns included in the state's price buffer.
    sentiment_series : pd.Series, optional
        Date-indexed sentiment score in [-1, 1] (Chapter 3.3). If not
        given, defaults to 0.0 (neutral) for every date -- see lm_sentiment.py
        to build a real one from ticker-aligned news text.
    use_sentiment : bool
        Chapter 5's ablation switch: True includes the sentiment scalar in
        the state vector (Table 3.1); False drops that dimension entirely
        (not just zeroes it) so the "without sentiment" arm is a genuinely
        smaller network, not the same network fed a constant.
    transaction_cost : float
        Proportional cost applied to daily turnover. Defaults to 0.0 to
        match the baselines (see Chapter 5.4's discussion of this
        simplification); set > 0 to experiment with a more realistic reward.
    """

    def __init__(self, tickers, start, end, prices=None, history_len=10,
                 sentiment_series=None, use_sentiment=True, transaction_cost=0.0):
        self.tickers = list(tickers)
        self.m = len(self.tickers)
        self.history_len = history_len
        self.use_sentiment = use_sentiment
        self.transaction_cost = transaction_cost

        if prices is None:
            import yfinance as yf
            prices = yf.download(self.tickers, start=start, end=end,
                                  threads=False, auto_adjust=True)["Close"]
        prices = prices[self.tickers].dropna()

        self.dates = prices.index
        self.returns = prices.pct_change().fillna(0.0)
        self.macd = prices.apply(compute_macd)
        self.rsi = prices.apply(compute_rsi)

        if sentiment_series is not None:
            self.sentiment = sentiment_series.reindex(self.dates).fillna(0.0)
        else:
            self.sentiment = pd.Series(0.0, index=self.dates)

        state_dim = history_len * self.m + 2 * self.m + self.m
        if use_sentiment:
            state_dim += 1
        self.state_dim = state_dim
        self.action_dim = self.m

        self.t = None
        self.weights = None

    def reset(self):
        self.t = self.history_len
        self.weights = np.full(self.m, 1.0 / self.m, dtype=np.float32)
        return self._get_state()

    def _get_state(self):
        hist = self.returns.iloc[self.t - self.history_len:self.t].values.flatten()
        tech = np.concatenate([self.macd.iloc[self.t].values, self.rsi.iloc[self.t].values])
        parts = [hist, tech, self.weights]
        if self.use_sentiment:
            parts.append(np.array([self.sentiment.iloc[self.t]]))
        return np.concatenate(parts).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), 0, None)
        action = action / (action.sum() + 1e-8)

        day_returns = self.returns.iloc[self.t].values
        gross_reward = float(np.dot(action, day_returns))
        turnover = float(np.abs(action - self.weights).sum())
        reward = gross_reward - self.transaction_cost * turnover

        date_of_return = self.dates[self.t]
        self.weights = action
        self.t += 1
        done = self.t >= len(self.dates) - 1
        next_state = None if done else self._get_state()
        info = {"date": date_of_return, "portfolio_return": gross_reward,
                "net_return": reward, "turnover": turnover}
        return next_state, reward, done, info
