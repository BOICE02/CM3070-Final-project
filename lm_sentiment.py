"""
Loughran-McDonald sentiment scoring (CM3020 Draft Report, Chapter 3.3).

Two pieces:
  1. load_lm_dictionary() -- loads the official Loughran-McDonald Master
     Dictionary. Download it from:
     https://sraf.nd.edu/loughranmcdonald-master-dictionary/
     and pass the CSV path in. Without a path, falls back to a tiny demo
     wordlist -- NOT the real dictionary, only good for testing the
     pipeline end-to-end before the real one is wired in.
  2. daily_sentiment_series() -- turns ticker-aligned news text into the
     date-indexed sentiment Series that portfolio_env.PortfolioEnv expects.

This module assumes NER ticker-alignment and sentence-level tokenisation
(Chapter 3.3) have already happened -- i.e. news_df already has one row per
(ticker, sentence), not raw multi-asset paragraphs.
"""
import re

import pandas as pd

# Demo-only wordlist. Replace with the real Loughran-McDonald dictionary
# (see module docstring) before running real experiments.
_DEMO_POSITIVE = {
    "gain", "gains", "growth", "profit", "profits", "strong", "outperform",
    "beat", "beats", "record", "improved", "improving", "upgrade", "surge",
    "rally", "exceeded", "robust", "optimistic",
}
_DEMO_NEGATIVE = {
    "loss", "losses", "decline", "declines", "weak", "miss", "misses",
    "downgrade", "litigation", "lawsuit", "recall", "bankruptcy", "plunge",
    "slump", "warning", "investigation", "layoffs",
}

_WORD_RE = re.compile(r"[A-Za-z']+")


def load_lm_dictionary(csv_path=None):
    """Returns (positive_words, negative_words) as lowercase sets."""
    if csv_path is None:
        return set(_DEMO_POSITIVE), set(_DEMO_NEGATIVE)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    positive = set(df.loc[df["Positive"] > 0, "Word"].str.lower())
    negative = set(df.loc[df["Negative"] > 0, "Word"].str.lower())
    return positive, negative


def score_text(text, positive_words, negative_words, scale=20.0):
    """
    Chapter 3.3: score = (pos_count - neg_count) / total_words, clipped to
    [-1, 1]. `scale` amplifies the raw ratio, which is otherwise tiny for
    short sentences (most words carry no sentiment), so realistic headlines
    don't all collapse to ~0.
    """
    words = [w.lower() for w in _WORD_RE.findall(str(text))]
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    raw = (pos - neg) / len(words)
    return max(-1.0, min(1.0, raw * scale))


def daily_sentiment_series(news_df, positive_words, negative_words,
                            date_col="date", text_col="text"):
    """
    news_df : one row per ticker-aligned news sentence, with at least
              [date_col, text_col]. If it also has a ticker column, filter
              to the basket's tickers before calling this (per-asset
              aggregation is a natural next step, but Table 3.1 currently
              treats V_t as a single scalar per day -- see Chapter 3.3).
    Returns : pandas Series indexed by date, mean sentiment score per day.
    """
    scores = news_df[text_col].apply(lambda t: score_text(t, positive_words, negative_words))
    daily = scores.groupby(pd.to_datetime(news_df[date_col])).mean()
    daily.index.name = "date"
    return daily
