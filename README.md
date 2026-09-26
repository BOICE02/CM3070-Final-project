# AI Financial Advisor Bot

CM3020 Artificial Intelligence — Final Project (Project Idea #2, "Financial Advisor Bot")
University of London, BSc Computer Science

A DDPG reinforcement-learning agent for active portfolio management over a fixed basket
of five large-cap US equities (AAPL, MSFT, GOOGL, AMZN, NVDA), combined with a
Loughran–McDonald financial-sentiment feature scored from real news headlines.

Full write-up, literature review, design rationale, and the complete evaluation
(including a 69-seed statistical ablation of the sentiment feature) are in the
accompanying report.

## Files

| File | Purpose |
|---|---|
| `portfolio_env.py` | The MDP environment (state, action, reward) |
| `ddpg_agent.py` | DDPG actor-critic implementation |
| `lm_sentiment.py` | Loughran–McDonald sentiment scoring |
| `load_news_data.py` | Loads real news headlines into a daily sentiment series |
| `train_agent.py` | Single training run |
| `train_multi_seed.py` | Trains both arms (with/without sentiment) across multiple seeds |
| `eval_metrics.py` | Sharpe ratio, drawdown, and other evaluation metrics |
| `backtest_harness.py` | Backtests the agent against equal-weight and S&P 500 baselines |
| `evaluate_multi_seed.py` | Aggregates multi-seed results with confidence intervals and a paired t-test |
| `app.py` | Streamlit web interface for non-technical users |
| `sentiment_series.csv` | Precomputed daily sentiment scores (2011–2020) from real news data |
| `multi_seed_results.csv` | Raw per-seed evaluation results (69 seeds × 2 arms × 2 periods) |

## Setup

```bash
pip install torch pandas numpy yfinance scipy streamlit
```

## Usage

Train both arms across several seeds:
```bash
python3 train_multi_seed.py --seeds 0 1 2 3 4
```

Evaluate with statistical comparison:
```bash
python3 evaluate_multi_seed.py --seeds 0 1 2 3 4
```

Run the web interface:
```bash
streamlit run app.py
```

## Headline result

Across 69 independently trained seeds per arm, the sentiment feature (as currently
implemented, using real but partial-coverage news data) shows a **regime-dependent**
effect on risk-adjusted return: no statistically significant effect in the volatile
2020 period (d≈0.03, p=0.77 — a confirmed null), but a small, statistically
significant benefit in the calmer 2019 period (d=0.27, 95% CI 0.03–0.58, p=0.029).

This finding only emerged after scaling from 5, to 10, to 20, to 69 seeds per arm —
smaller samples suggested a larger effect in both periods that did not hold up. That
trajectory is discussed as a methodological point in its own right, connected to
Henderson et al. (2018) "Deep Reinforcement Learning that Matters" on seed-variance
risk in DRL evaluation. See the report's Evaluation chapter (Section 5.7) for the
full analysis.
