import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(
    page_title="Stock Market Tracker",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Tracker")
st.caption("Interactive stock analysis | Data from Yahoo Finance via yfinance")

# 1. Select stocks
st.sidebar.header("Stock Settings")

symbols_text = st.sidebar.text_input(
    "Enter stock symbols (separated by commas)",
    "AAPL, NVDA, TSLA, MSFT, AMZN"
)

symbols = list(dict.fromkeys(
    symbol.strip().upper()
    for symbol in symbols_text.split(",")
    if symbol.strip()
))

# 2. Select date range
today = date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    today - timedelta(days=90)
)

end_date = st.sidebar.date_input(
    "End Date",
    today
)

if not symbols:
    st.warning("Please enter at least one stock symbol.")
    st.stop()

if start_date >= end_date:
    st.error("Start Date must be earlier than End Date.")
    st.stop()

# 3. Download stock data
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    return yf.download(
        tickers=list(tickers),
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        threads=False
    )

with st.spinner("Loading stock data..."):
    try:
        data = load_data(tuple(symbols), start_date, end_date)
    except Exception as e:
        st.error(f"Could not load stock data: {e}")
        st.stop()

if data.empty:
    st.warning("No stock data found. Check the symbols and dates.")
    st.stop()

# 4. Extract closing prices and volume
prices = data["Close"].copy()
volumes = data["Volume"].copy()

if isinstance(prices, pd.Series):
    prices = prices.to_frame(name=symbols[0])
    volumes = volumes.to_frame(name=symbols[0])

prices = prices.dropna(axis=1, how="all")

if prices.empty:
    st.warning("No valid closing prices were found.")
    st.stop()

# 5. Calculate performance
normalized = prices.apply(
    lambda col: col / col.dropna().iloc[0] * 100
    if not col.dropna().empty
    else col
)

daily_returns = prices.pct_change(fill_method=None) * 100

# 6. Display chart helper
def show_chart(frame, title, y_label):
    fig = go.Figure()

    for symbol in frame.columns:
        fig.add_trace(go.Scatter(
            x=frame.index,
            y=frame[symbol],
            mode="lines",
            name=symbol
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        template="plotly_dark",
        hovermode="x unified",
        height=520
    )

    st.plotly_chart(fig, use_container_width=True)

# 7. Show analysis tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Stock Prices",
    "Normalized Performance",
    "Daily Returns",
    "Trading Volume"
])

with tab1:
    show_chart(prices, "Stock Prices", "Price (USD)")

with tab2:
    show_chart(
        normalized,
        "Normalized Stock Performance (Start = 100)",
        "Normalized Price"
    )

with tab3:
    show_chart(
        daily_returns,
        "Daily Percentage Change",
        "Daily Return (%)"
    )

with tab4:
    show_chart(
        volumes.reindex(columns=prices.columns),
        "Trading Volume",
        "Shares Traded"
    )

# 8. Performance summary
st.subheader("Performance Summary")

summary = pd.DataFrame({
    "First Price": prices.apply(lambda col: col.dropna().iloc[0]),
    "Latest Price": prices.apply(lambda col: col.dropna().iloc[-1]),
    "Total Return (%)": prices.apply(
        lambda col: (
            col.dropna().iloc[-1] / col.dropna().iloc[0] - 1
        ) * 100
    )
})

st.dataframe(summary.round(2), use_container_width=True)

st.caption(
    "Market data may be delayed or incomplete. "
    "For educational and research purposes only."
)
