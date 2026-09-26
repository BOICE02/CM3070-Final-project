"""
Portfolio evaluation metrics for the AI Financial Advisor Bot backtests
(CM3020 Draft Report, Chapter 5).

Feed each strategy's daily returns (a pandas Series) into summarize() to
get the numbers that go straight into the Chapter 5 results table.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cumulative_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns).prod() - 1


def annualized_return(daily_returns: pd.Series) -> float:
    n = len(daily_returns)
    if n == 0:
        return np.nan
    years = n / TRADING_DAYS
    total = cumulative_return(daily_returns)
    return (1 + total) ** (1 / years) - 1 if years > 0 else np.nan


def annualized_volatility(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(TRADING_DAYS)


def max_drawdown(daily_returns: pd.Series) -> float:
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()


def summarize(daily_returns: pd.Series, label: str = "") -> dict:
    """One row for the Chapter 5 results table."""
    if isinstance(daily_returns, pd.DataFrame):
        daily_returns = daily_returns.iloc[:, 0]  # guard against a stray 1-col DataFrame
    return {
        "strategy": label,
        "cumulative_return": cumulative_return(daily_returns),
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(daily_returns),
    }
