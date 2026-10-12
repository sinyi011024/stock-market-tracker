import io
import re
from datetime import date, timedelta
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Equity Atlas | U.S. Stocks", page_icon="📈", layout="wide")

COLORS = ["#688B73", "#C47D5A", "#6687B0", "#9A78A5", "#C3A14E",
          "#477D85", "#BD7286", "#897A63", "#557FBA", "#A77C54",
          "#6C987F", "#B26F64", "#7776AD", "#9C9252", "#4E8F9B"]

st.markdown("""
<style>
:root {color-scheme: light;}
.stApp, [data-testid="stAppViewContainer"] {background:#F7F5F1; color:#35423C;}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {background:#EEEFE9 !important; color:#35423C !important;}
.stApp p, .stApp label, .stApp span, .stApp small,
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {color:#35423C;}
[data-testid="stMarkdownContainer"] p {color:#35423C;}
[data-testid="stCaptionContainer"] p, [data-testid="stWidgetLabel"] p {color:#65746C !important;}
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E4E5DE;
    border-radius:14px; padding:16px;}
[data-testid="stMetricLabel"] *, [data-testid="stMetricValue"] * {color:#35423C !important;}
[data-testid="stMetricValue"] {color:#35423C !important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#E3E5DE;}
h1,h2,h3 {color:#35423C !important; letter-spacing:-.02em;}
[data-baseweb="input"] > div, [data-baseweb="select"] > div,
[data-baseweb="base-input"] {background:#FFFFFF !important; color:#35423C !important;
 border-color:#CBD3CA !important;}
[data-baseweb="input"] input, [data-baseweb="select"] input,
[data-baseweb="base-input"] input {color:#35423C !important; -webkit-text-fill-color:#35423C !important;}
[data-baseweb="input"] input::placeholder {color:#78867D !important;}
[data-baseweb="select"] *, [data-baseweb="tag"] * {color:#35423C;}
[data-baseweb="tag"] {background:#DCE8DD !important;}
[data-baseweb="tag"] span {color:#35423C !important;}
[data-baseweb="radio"] label, [data-baseweb="radio"] span {color:#35423C !important;}
.stButton > button[kind="primary"] {background:#688B73; border-color:#688B73;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_listings():
    """Official Nasdaq Trader directory: Nasdaq + other exchange listings.
    Directory may be temporarily unavailable; manual ticker input still works.
    """
    endpoints = [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol", "Security Name", "NASDAQ"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol", "Security Name", "Other U.S. exchange"),
    ]
    frames = []
    for url, ticker_col, name_col, exchange in endpoints:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=12) as response:
                content = response.read().decode("utf-8-sig", errors="replace")
            data = pd.read_csv(io.StringIO(content), sep="|", dtype=str)
            data = data[data[ticker_col].notna()].copy()
            data = data[~data[ticker_col].str.contains("File Creation Time", case=False, na=False)]
            if "Test Issue" in data.columns:
                data = data[data["Test Issue"].fillna("N") != "Y"]
            data = data[[ticker_col, name_col]].rename(
                columns={ticker_col: "Ticker", name_col: "Company"})
            data["Exchange"] = exchange
            frames.append(data)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["Ticker", "Company", "Exchange"])
    listings = pd.concat(frames, ignore_index=True)
    listings["Ticker"] = listings["Ticker"].str.strip().str.upper()
    listings["Company"] = listings["Company"].fillna("").str.strip()
    listings = listings.drop_duplicates(subset="Ticker").sort_values("Ticker")
    return listings.reset_index(drop=True)

@st.cache_data(ttl=900, show_spinner=False)
def get_prices(tickers, period, start_date, end_date):
    options = dict(tickers=list(tickers), auto_adjust=True, progress=False,
                   threads=True, group_by="column")
    periods = {
        "1D": ("1d", "5m"), "5D": ("5d", "30m"),
        "1M": ("1mo", "1d"), "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"), "1Y": ("1y", "1d"),
        "3Y": ("3y", "1d"), "5Y": ("5y", "1d"),
        "10Y": ("10y", "1d"), "MAX": ("max", "1d"),
    }
    if period == "Custom":
        options.update(start=str(start_date), end=str(end_date + timedelta(days=1)),
                       interval="1d")
    else:
        yf_period, interval = periods[period]
        # Yahoo Finance does not consistently support 3y as a period string.
        if period == "3Y":
            options.update(start=str(date.today() - timedelta(days=3*365+2)),
                           end=str(date.today() + timedelta(days=1)), interval="1d")
        else:
            options.update(period=yf_period, interval=interval)
    raw = yf.download(**options)
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.sort_index().dropna(axis=1, how="all")

st.title("Equity Atlas")
st.caption("U.S. Equity Analytics  ·  Search, compare and explore historical stock performance")

listings = get_us_listings()
with st.sidebar:
    st.header("Explore stocks")
    if not listings.empty:
        st.caption(f"U.S. listing directory: {len(listings):,} symbols (includes ETFs and other securities).")
        query = st.text_input("Search company name or ticker", placeholder="e.g. Costco, PLTR, JPM")
        if query.strip():
            matches = listings[
                listings["Ticker"].str.contains(re.escape(query.strip()), case=False, na=False)
                | listings["Company"].str.contains(re.escape(query.strip()), case=False, na=False)
            ].head(40)
            labels = [f"{r.Ticker} — {r.Company}" for r in matches.itertuples()]
            picks = st.multiselect("Search results (up to 40 shown)", labels)
            selected_search = [label.split(" — ", 1)[0] for label in picks]
        else:
            selected_search = []
            st.caption("Search by name to find companies in the U.S. exchange directory.")
    else:
        selected_search = []
        st.info("Exchange directory temporarily unavailable. You can still enter any ticker below.")

    manual = st.text_input("Ticker symbols (comma-separated)",
                           value="AAPL, NVDA, MSFT, TSLA, AMZN",
                           help="You can enter any supported U.S. ticker, even if directory search is unavailable.")
    symbols = list(dict.fromkeys(
        selected_search + [s.strip().upper() for s in manual.split(",") if s.strip()]
    ))
    invalid = [s for s in symbols if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", s)]
    symbols = [s for s in symbols if s not in invalid]
    if invalid:
        st.warning("Unsupported symbol format: " + ", ".join(invalid))

    period = st.selectbox("Historical range",
        ["1D", "5D", "1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MAX", "Custom"],
        index=8)
    start = end = None
    if period == "Custom":
        start = st.date_input("Start date", value=date.today()-timedelta(days=3650),
                              max_value=date.today())
        end = st.date_input("End date (inclusive)", value=date.today(),
                            max_value=date.today())
    mode = st.radio("Chart", ["Normalized performance", "Adjusted stock price"],
                    horizontal=False)
    st.caption("1D / 5D use intraday data; long-range charts use daily adjusted closes.")

if not symbols:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()
if len(symbols) > 25:
    st.warning("Please compare up to 25 stocks at once. The searchable directory has no five-stock limit.")
    st.stop()
if period == "Custom" and start > end:
    st.error("Start date must be before or equal to end date.")
    st.stop()

with st.spinner("Fetching historical market data..."):
    try:
        prices = get_prices(tuple(symbols), period, start, end)
    except Exception as exc:
        st.error(f"Could not retrieve prices: {exc}")
        st.stop()
if prices.empty:
    st.error("No prices returned. Check the symbols or date range, or retry later.")
    st.stop()

missing = [s for s in symbols if s not in prices.columns or prices[s].dropna().empty]
if missing:
    st.warning("No available price data: " + ", ".join(missing))
prices = prices.dropna(axis=1, how="all")
if prices.empty:
    st.stop()

returns = {}
for ticker in prices.columns:
    s = prices[ticker].dropna()
    if len(s) >= 2 and s.iloc[0] != 0:
        returns[ticker] = (s.iloc[-1] / s.iloc[0] - 1) * 100

m1, m2, m3 = st.columns(3)
m1.metric("Selected stocks", len(prices.columns))
m2.metric("History", period)
m3.metric("Top return in range", f"{max(returns, key=returns.get)}: {max(returns.values()):+.1f}%"
          if returns else "N/A")


st.subheader("Stock Performance Overview")

st.caption(f"{len(prices.columns)} stocks · {period} historical performance")

for start_idx in range(0, len(prices.columns), 3):
    tickers_row = list(prices.columns)[start_idx:start_idx + 3]
    cols = st.columns(3)

    for i, ticker in enumerate(tickers_row):
        stock_prices = prices[ticker].dropna()

        if stock_prices.empty:
            continue

        first_price = stock_prices.iloc[0]
        latest_price = stock_prices.iloc[-1]

        total_return = (
            (latest_price / first_price - 1) * 100
            if first_price != 0 else float("nan")
        )

        with cols[i]:
            st.metric(
                label=f"{ticker} · {period}",
                value=f"${latest_price:,.2f}",
                delta=f"{total_return:+.2f}%"
            )

st.subheader("Performance Comparison")

fig = go.Figure()

for i, ticker in enumerate(prices.columns):
    stock_prices = prices[ticker].dropna()

    if stock_prices.empty:
        continue

    if mode == "Normalized performance":
        chart_data = stock_prices / stock_prices.iloc[0] * 100
    else:
        chart_data = stock_prices

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data.values,
            mode="lines",
            name=ticker,
            line=dict(
                color=COLORS[i % len(COLORS)],
                width=2.5
            )
        )
    )

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    height=520,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title=(
        "Index (Start = 100)"
        if mode == "Normalized performance"
        else "Adjusted Price (USD)"
    ),
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig, use_container_width=True)
