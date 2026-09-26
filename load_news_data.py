"""
Turns the Kaggle "Daily Financial News for 6000+ Stocks" dataset into the
date-indexed sentiment Series that PortfolioEnv expects (Chapter 3.3),
using lm_sentiment.py's scoring.

Download the dataset first (free Kaggle account required):
    https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
It ships as one or two CSVs (commonly raw_analyst_ratings.csv and/or
raw_partner_headlines.csv). Column names have varied across re-uploads of
this dataset, so this script auto-detects the date/headline/ticker columns
rather than hard-coding them -- check the printed "Detected columns" line
against the actual file if anything looks wrong.

Usage:
    python3 load_news_data.py --csv raw_analyst_ratings.csv raw_partner_headlines.csv analyst_ratings_processed.csv
    python3 load_news_data.py --csv raw_analyst_ratings.csv --lm-dict LoughranMcDonald_MasterDictionary.csv

Output:
    sentiment_series.csv   -- columns [date, sentiment_score], one row per
                               date with at least one basket-ticker headline.
                               train_multi_seed.py / evaluate_multi_seed.py
                               load this automatically if present.
"""
import argparse

import pandas as pd

from lm_sentiment import load_lm_dictionary, daily_sentiment_series

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

_DATE_CANDIDATES = ["date", "Date", "published_date", "datetime"]
_TEXT_CANDIDATES = ["headline", "title", "text", "Headline", "Title"]
_TICKER_CANDIDATES = ["stock", "ticker", "symbol", "Stock", "Ticker", "Symbol"]


def _detect_column(df, candidates, purpose):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not auto-detect the {purpose} column. Available columns: "
        f"{list(df.columns)}. Pass --{purpose}-col explicitly."
    )


def load_and_filter(csv_path, tickers, date_col=None, text_col=None, ticker_col=None):
    df = pd.read_csv(csv_path)
    date_col = date_col or _detect_column(df, _DATE_CANDIDATES, "date")
    text_col = text_col or _detect_column(df, _TEXT_CANDIDATES, "text")
    ticker_col = ticker_col or _detect_column(df, _TICKER_CANDIDATES, "ticker")
    print(f"Detected columns: date={date_col!r}, text={text_col!r}, ticker={ticker_col!r}")

    df = df[[date_col, text_col, ticker_col]].rename(
        columns={date_col: "date", text_col: "text", ticker_col: "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    before = len(df)
    df = df[df["ticker"].isin(tickers)].dropna(subset=["date", "text"])
    print(f"Filtered {before} rows -> {len(df)} rows for tickers {tickers}")

    # Dates in this dataset are sometimes full timestamps with a trailing
    # timezone or time-of-day, and sometimes bare dates -- format='mixed' is
    # required or pandas silently turns every row after the first format
    # switch into NaT (verified while testing this script; without it, rows
    # were dropped with no warning, not just reformatted).
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date"])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs="+",
                         help="Path(s) to the downloaded Kaggle CSV(s). The dataset ships as three "
                              "files (raw_analyst_ratings.csv, raw_partner_headlines.csv, "
                              "analyst_ratings_processed.csv) with complementary coverage -- "
                              "pass all three to maximise how many headlines are found.")
    parser.add_argument("--lm-dict", default=None,
                         help="Path to the official Loughran-McDonald Master Dictionary CSV. "
                              "Omit to fall back to the small demo wordlist (not for real results).")
    parser.add_argument("--date-col", default=None)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--ticker-col", default=None)
    parser.add_argument("--out", default="sentiment_series.csv")
    args = parser.parse_args()

    frames = []
    for csv_path in args.csv:
        print(f"\nLoading {csv_path} ...")
        frames.append(load_and_filter(csv_path, TICKERS, args.date_col, args.text_col, args.ticker_col))
    news_df = pd.concat(frames, ignore_index=True)
    before_dedup = len(news_df)
    news_df = news_df.drop_duplicates(subset=["date", "text", "ticker"])
    print(f"\nCombined {len(args.csv)} file(s): {before_dedup} rows -> {len(news_df)} after de-duplication")
    print("Per-ticker headline counts:")
    print(news_df["ticker"].value_counts().reindex(TICKERS, fill_value=0))

    if news_df.empty:
        print("No matching headlines found for the basket tickers -- check the ticker "
              "column values against", TICKERS)
        return

    positive_words, negative_words = load_lm_dictionary(args.lm_dict)
    if args.lm_dict is None:
        print("WARNING: using the small demo wordlist, not the official Loughran-McDonald "
              "dictionary. Pass --lm-dict for real results (Chapter 4.3's known limitation).")

    series = daily_sentiment_series(news_df, positive_words, negative_words)
    series.to_frame("sentiment_score").to_csv(args.out)
    print(f"Saved {len(series)} daily sentiment values to {args.out}")
    print(f"Date range: {series.index.min().date()} to {series.index.max().date()}")
    print(f"Mean={series.mean():.4f}, non-zero days={int((series != 0).sum())}/{len(series)}")


if __name__ == "__main__":
    main()
