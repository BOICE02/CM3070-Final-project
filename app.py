"""
AI Financial Advisor Bot — Streamlit interface (CM3020 Final Report, Chapter 3.1
"interface phase" / Chapter 1's non-technical-user requirement).

Run:
    pip install streamlit
    streamlit run app.py

Expects, in the same folder:
    - portfolio_env.py, ddpg_agent.py            (already have these)
    - ddpg_actor_with_sentiment.pt                (from train_agent.py --sentiment)
    - ddpg_actor_no_sentiment.pt                  (from train_agent.py --no-sentiment)
    - results_2020_volatile.csv, results_2019_calm.csv     (from backtest_harness.py)
    - equity_curve_2020_volatile.png, equity_curve_2019_calm.png  (from backtest_harness.py)

Any of the above that are missing degrade gracefully (a clear on-screen message
instead of a crash) so the rest of the app still works.
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from portfolio_env import PortfolioEnv
from ddpg_agent import DDPGAgent

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

st.set_page_config(page_title="AI Financial Advisor Bot", page_icon="\U0001F916", layout="wide")


@st.cache_data(ttl=3600)
def fetch_recent_prices(days: int = 90) -> pd.DataFrame:
    import yfinance as yf
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=days)
    prices = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                          threads=False, auto_adjust=True)["Close"]
    return prices[TICKERS].dropna()


def get_recommendation(prices: pd.DataFrame, weights_path: str, use_sentiment: bool):
    """
    Builds the state for the most recent available day (Table 3.1) and asks
    the trained actor for a recommended allocation. Assumes the user is
    currently holding an equal-weight position (a simplification for this
    demo interface -- a real deployment would take the user's actual
    holdings as input instead).
    """
    env = PortfolioEnv(TICKERS, None, None, prices=prices, use_sentiment=use_sentiment)
    env.t = len(env.dates) - 1
    env.weights = np.full(env.m, 1.0 / env.m, dtype=np.float32)
    state = env._get_state()

    agent = DDPGAgent(env.state_dim, env.action_dim)
    agent.load(weights_path)
    weights = agent.act(state, noise_scale=0.0)
    return dict(zip(TICKERS, weights)), env


def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


st.title("\U0001F916 AI Financial Advisor Bot")
st.caption("A DDPG reinforcement-learning agent for a fixed basket of five large-cap US equities \u2014 CM3020 Final Report")
st.warning("This is a research prototype for a university project, not financial advice.")

with st.sidebar:
    st.header("Settings")
    sentiment_choice = st.radio("Agent version", ["With sentiment", "Without sentiment"])
    use_sentiment = sentiment_choice == "With sentiment"
    st.markdown("---")
    st.markdown("**Asset basket:** " + ", ".join(TICKERS))
    st.markdown("**Algorithm:** DDPG actor-critic (Chapter 3\u20134)")
    st.caption("See Chapter 5.8 of the report for a full critique of this project, "
               "including its limitations.")

weights_path = "ddpg_actor_with_sentiment.pt" if use_sentiment else "ddpg_actor_no_sentiment.pt"

st.header("Today's Recommended Portfolio")
try:
    prices = fetch_recent_prices()
    rec, env = get_recommendation(prices, weights_path, use_sentiment)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots()
        ax.pie(list(rec.values()), labels=list(rec.keys()), autopct="%1.1f%%", startangle=90)
        ax.set_title("Recommended allocation")
        st.pyplot(fig)
    with col2:
        st.subheader("Why this allocation?")
        latest_macd = env.macd.iloc[-1]
        latest_rsi = env.rsi.iloc[-1]
        explain_df = pd.DataFrame({
            "Weight": [f"{rec[t] * 100:.1f}%" for t in TICKERS],
            "MACD signal": [f"{latest_macd[t]:+.3f}" for t in TICKERS],
            "RSI signal": [f"{latest_rsi[t]:+.2f}" for t in TICKERS],
        }, index=TICKERS)
        st.dataframe(explain_df, use_container_width=True)
        st.info("Sentiment score: 0.00 (neutral) for every asset \u2014 no live news feed is "
                "connected yet in this prototype; see Chapter 3.3 / 4.3 of the report.")
except FileNotFoundError:
    arm = "--sentiment" if use_sentiment else "--no-sentiment"
    st.error(f"Trained weights not found ({weights_path}). Run `python3 train_agent.py {arm}` first, "
             "then reload this page.")
except Exception as e:
    st.error(f"Could not fetch live data or generate a recommendation right now: {e}")

st.header("Backtested Performance (Chapter 5)")
period = st.selectbox("Period", ["2020 (volatile)", "2019 (calmer)"])
period_key = "2020_volatile" if period.startswith("2020") else "2019_calm"

results = load_csv(f"results_{period_key}.csv")
if results is not None:
    st.dataframe(
        results.set_index("strategy").style
        .format("{:.2%}", subset=["cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown"])
        .format("{:.2f}", subset=["sharpe_ratio"]),
        use_container_width=True,
    )
else:
    st.warning(f"results_{period_key}.csv not found \u2014 run backtest_harness.py for this period first.")

try:
    st.image(f"equity_curve_{period_key}.png", caption=f"Equity curves \u2014 {period}")
except Exception:
    st.warning(f"equity_curve_{period_key}.png not found yet.")

st.caption("AI Financial Advisor Bot \u2014 CM3020 Final Report. Not financial advice.")
