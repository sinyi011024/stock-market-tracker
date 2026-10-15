import io
import re
from datetime import date, timedelta
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Equity Atlas | U.S. Stocks", page_icon="📈", layout="wide")

COLORS = ["#688B73", "#C47D5A", "#6687B0", "#9A78A5", "#C3A14E",
          "#477D85", "#BD7286", "#897A63", "#557FBA", "#A77C54",
          "#6C987F", "#B26F64", "#7776AD", "#9C9252", "#4E8F9B"]
PROFILES = {"Conservative": (0.25, 0.60), "Balanced": (0.40, 0.85), "Aggressive": (0.60, 1.00)}
INTRADAY = {"1D": ("1d", "5m"), "5D": ("5d", "30m")}
LOOKBACK = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "3Y": 1097, "5Y": 1827, "10Y": 3652}
RANGES = ["1D", "5D", "1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MAX", "Custom"]
DEFAULT_W = {"w_trend": 30, "w_cross": 20, "w_mom": 35, "w_rsi": 15}
DET_DAYS = {"3M": 92, "6M": 183, "1Y": 366, "2Y": 731, "5Y": 1827}
METRIC_FMT = {"Total Return %": "{:+.1f}%", "Annualized Return %": "{:+.1f}%",
              "Annualized Volatility %": "{:.1f}%", "Sharpe": "{:.2f}", "Max Drawdown %": "{:.1f}%"}

@st.cache_data(ttl=86400, show_spinner=False)
def get_us_listings():
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
            data = data[[ticker_col, name_col]].rename(columns={ticker_col: "Ticker", name_col: "Company"})
            data["Exchange"] = exchange
            frames.append(data)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["Ticker", "Company", "Exchange"])
    listings = pd.concat(frames, ignore_index=True)
    listings["Ticker"] = listings["Ticker"].str.strip().str.upper().str.replace(".", "-", regex=False)
    listings["Company"] = listings["Company"].fillna("").str.strip()
    listings = listings.drop_duplicates(subset="Ticker").sort_values("Ticker")
    return listings.reset_index(drop=True)


def _close(raw, tickers):
    if raw.empty:
        return pd.DataFrame()
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    return close.sort_index().dropna(axis=1, how="all")


@st.cache_data(ttl=900, show_spinner=False)
def get_daily(tickers, start):
    raw = yf.download(list(tickers), start=start, end=str(date.today() + timedelta(days=1)),
                      interval="1d", auto_adjust=True, progress=False, threads=True, group_by="column")
    return _close(raw, tickers)


@st.cache_data(ttl=900, show_spinner=False)
def get_intraday(tickers, period):
    yf_period, interval = INTRADAY[period]
    raw = yf.download(list(tickers), period=yf_period, interval=interval, auto_adjust=True,
                      progress=False, threads=True, group_by="column")
    return _close(raw, tickers)


def ret(p):
    return p.pct_change(fill_method=None)


def calc_rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def indicators(p, wts=(0.30, 0.20, 0.35, 0.15)):
    ma50 = p.rolling(50).mean()
    ma200 = p.rolling(200).mean()
    mom = p / p.shift(126) - 1
    rsi = p.apply(calc_rsi)
    vol = ret(p).rolling(252, min_periods=126).std() * np.sqrt(252)
    t1 = (p > ma200).astype(float) * 2 - 1
    t2 = (ma50 > ma200).astype(float) * 2 - 1
    m = (mom / 0.3).clip(-1, 1)
    r = (rsi < 30).astype(float) - (rsi > 70).astype(float)
    score = (wts[0] * t1 + wts[1] * t2 + wts[2] * m + wts[3] * r) / (sum(wts) or 1)
    ok = ma200.notna() & mom.notna() & rsi.notna() & vol.notna()
    return ma200, mom, rsi, vol, score.where(ok)


def allocate(score, vol, cap, mx):
    n = len(score)
    pos = score.clip(lower=0)
    raw = pos / vol
    if n == 0 or raw.sum() <= 0:
        return pd.Series(0.0, index=score.index)
    exposure = min(pos.sum() / n, mx)
    w = raw / raw.sum() * exposure
    return w.clip(upper=max(cap, 1 / n))


def perf(s, rf):
    r = ret(s).dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] / s.iloc[0] - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else np.nan
    mdd = (s / s.cummax() - 1).min()
    return {"Total Return %": total * 100, "Annualized Return %": cagr * 100,
            "Annualized Volatility %": vol * 100, "Sharpe": sharpe, "Max Drawdown %": mdd * 100}


def backtest(p, cap, mx, rf, wts):
    _, _, _, vol, score = indicators(p, wts)
    idx = p.index
    reb = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).last().values
    W = pd.DataFrame(np.nan, index=idx, columns=p.columns)
    first = None
    for d in reb:
        sc = score.loc[d].dropna()
        if sc.empty:
            continue
        if first is None:
            first = d
        W.loc[d] = allocate(sc, vol.loc[d, sc.index], cap, mx).reindex(p.columns).fillna(0)
    if first is None:
        return None
    W = W.ffill().fillna(0)
    Wl = W.shift(1).fillna(0)
    rets = ret(p)
    port = (Wl * rets.fillna(0)).sum(axis=1) + (1 - Wl.sum(axis=1)) * rf / 252
    ew = rets.mean(axis=1)
    return port.loc[first:], ew.loc[first:]


def simulate(px, amount, dca):
    buys = pd.Series(0.0, index=px.index)
    if dca:
        first = px.index.to_series().groupby([px.index.year, px.index.month]).first()
        idx = pd.DatetimeIndex(first.values)
        buys.loc[idx] = (amount / px.loc[idx]).values
    else:
        buys.iloc[0] = amount / px.iloc[0]
    return buys.cumsum() * px, (buys > 0).cumsum() * amount


def reset_weights():
    for k, v in DEFAULT_W.items():
        st.session_state[k] = v


def style(fig, ytitle="", height=520):
    fig.update_layout(
        height=height, margin=dict(l=55, r=30, t=30, b=80), hovermode="x unified",
        xaxis=dict(title="Date"), yaxis=dict(title=ytitle),
        legend=dict(orientation="h", y=-0.22, x=0),
    )
    return fig


def chart_start(period, start):
    if period == "Custom":
        return start
    if period == "MAX":
        return date(1970, 1, 2)
    return date.today() - timedelta(days=LOOKBACK[period])


st.title("Equity Atlas")
st.caption("U.S. Equity Analytics  ·  Search, compare and explore historical stock performance")

listings = get_us_listings()
names = {r.Ticker: re.split(r" - | Common | Class ", r.Company)[0].strip()
         for r in listings.itertuples()}
st.session_state.setdefault("picked", [])

with st.sidebar:
    st.header("Explore stocks")
    matches = []
    if not listings.empty:
        st.caption(f"U.S. listing directory: {len(listings):,} symbols (includes ETFs and other securities).")
        query = st.text_input("Search company name or ticker", placeholder="e.g. Costco, PLTR, JPM")
        if query.strip():
            q = re.escape(query.strip())
            found = listings[
                listings["Ticker"].str.contains(q, case=False, na=False)
                | listings["Company"].str.contains(q, case=False, na=False)
            ].head(40)
            matches = [f"{r.Ticker} — {r.Company}" for r in found.itertuples()]
    else:
        st.info("Exchange directory temporarily unavailable. You can still enter any ticker below.")

    with st.form("controls"):
        options = list(dict.fromkeys(st.session_state.picked + matches))
        picks = st.multiselect("Search results (up to 40 shown)", options, default=st.session_state.picked)
        manual = st.text_input("Ticker symbols (comma-separated)",
                               value="AAPL, NVDA, MSFT, TSLA, AMZN",
                               help="You can enter any supported U.S. ticker.")
        period = st.selectbox("Historical range", RANGES, index=8)
        start = st.date_input("Custom start (used when range = Custom)",
                              value=date.today() - timedelta(days=3650), max_value=date.today())
        end = st.date_input("Custom end (inclusive)", value=date.today(), max_value=date.today())
        mode = st.radio("Chart", ["Normalized performance", "Adjusted stock price"])
        align = st.checkbox("Align start to common first day", value=True)
        show_bench = st.checkbox("Show SPY benchmark", value=True)
        profile = st.radio("Risk profile", list(PROFILES), index=1)
        capital = st.number_input("Capital (USD)", min_value=100.0, value=10000.0, step=500.0)
        rf = st.number_input("Risk-free rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25) / 100
        thr = st.slider("Signal threshold", 0.10, 0.60, 0.35, 0.05)
        st.form_submit_button("Update", type="primary")
    st.session_state.picked = picks
    st.caption("1D / 5D use intraday data; other ranges use daily adjusted closes.")

selected_search = [label.split(" — ", 1)[0] for label in picks]
symbols = list(dict.fromkeys(selected_search + [s.strip().upper().replace(".", "-")
                                                for s in manual.split(",") if s.strip()]))
invalid = [s for s in symbols if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", s)]
symbols = [s for s in symbols if s not in invalid]
if invalid:
    st.sidebar.warning("Unsupported symbol format: " + ", ".join(invalid))

if not symbols:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()
if len(symbols) > 25:
    st.warning("Please compare up to 25 stocks at once.")
    st.stop()
if period == "Custom" and start > end:
    st.error("Start date must be before or equal to end date.")
    st.stop()

fetch = tuple(dict.fromkeys(symbols + ["SPY"]))
cs = chart_start(period, start)
five_years = date.today() - timedelta(days=5 * 365 + 30)
hist_start = five_years if period in INTRADAY else min(cs, five_years)

with st.spinner("Fetching market data..."):
    try:
        hist = get_daily(fetch, str(hist_start))
        if hist.empty:
            st.error("No prices returned. Check the symbols or retry later.")
            st.stop()
        if period in INTRADAY:
            view = get_intraday(fetch, period)
        else:
            view = hist.loc[pd.Timestamp(cs):]
            if period == "Custom":
                view = view.loc[:pd.Timestamp(end)]
    except Exception as exc:
        st.error(f"Could not retrieve prices: {exc}")
        st.stop()

if view.empty:
    st.error("No prices returned for this range.")
    st.stop()

stocks = view[[s for s in symbols if s in view.columns]].dropna(axis=1, how="all")
missing = [s for s in symbols if s not in stocks.columns]
if missing:
    st.warning("No available price data: " + ", ".join(missing))
if stocks.empty:
    st.stop()
bench = view["SPY"].dropna() if "SPY" in view.columns else None

tab_ov, tab_cmp, tab_risk, tab_sig, tab_det, tab_wi, tab_bt = st.tabs(
    ["Overview", "Comparison", "Risk & Correlation", "Signals & Allocation",
     "Stock Detail", "What-if", "Backtest"])

with tab_ov:
    rets = {}
    for t in stocks.columns:
        s = stocks[t].dropna()
        if len(s) >= 2 and s.iloc[0] != 0:
            rets[t] = (s.iloc[-1] / s.iloc[0] - 1) * 100
    a, b, c = st.columns(3)
    a.metric("Selected stocks", len(stocks.columns))
    b.metric("History", period)
    c.metric("Top return in range",
             f"{max(rets, key=rets.get)}: {max(rets.values()):+.1f}%" if rets else "N/A")
    tick = list(stocks.columns)
    for i in range(0, len(tick), 3):
        for col, t in zip(st.columns(3), tick[i:i + 3]):
            s = stocks[t].dropna()
            if len(s) < 2:
                continue
            with col:
                st.metric(f"{t} · {period}", f"${s.iloc[-1]:,.2f}", f"{rets.get(t, float('nan')):+.2f}%")
                st.caption(names.get(t, ""))

with tab_cmp:
    cp = stocks.dropna() if align else stocks
    if cp.empty:
        cp = stocks
    normalized = mode == "Normalized performance"
    fig = go.Figure()
    for i, t in enumerate(cp.columns):
        s = cp[t].dropna()
        if s.empty:
            continue
        y = s / s.iloc[0] * 100 if normalized else s
        fig.add_trace(go.Scatter(
            x=y.index, y=y.values, mode="lines", name=t,
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra>%{fullData.name}</extra>"))
    if show_bench and bench is not None and "SPY" not in cp.columns:
        bs = bench.loc[cp.index[0]:].dropna()
        if len(bs) > 1:
            y = bs / bs.iloc[0] * 100 if normalized else bs
            fig.add_trace(go.Scatter(
                x=y.index, y=y.values, mode="lines", name="SPY",
                line=dict(color="#8A8F98", width=2, dash="dash"),
                hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}<extra>SPY</extra>"))
    style(fig, "Index (Start = 100)" if normalized else "Adjusted Price (USD)")
    st.plotly_chart(fig, width="stretch", key="stock_comparison_chart")

with tab_risk:
    if period in INTRADAY:
        st.info("Risk metrics and correlation need daily data. Choose 1M or longer.")
    else:
        rows = {t: perf(stocks[t].dropna(), rf) for t in stocks.columns if stocks[t].dropna().shape[0] > 2}
        if show_bench and bench is not None and len(bench) > 2 and "SPY" not in rows:
            rows["SPY"] = perf(bench, rf)
        if rows:
            mdf = pd.DataFrame(rows).T
            mdf.index.name = "Ticker"
            st.subheader("Risk & Return Metrics")
            st.dataframe(mdf.style.format(METRIC_FMT), width="stretch")
            st.download_button("Download metrics CSV", mdf.to_csv().encode(), "metrics.csv", "text/csv")
        if stocks.shape[1] >= 2:
            st.subheader("Daily Return Correlation")
            corr = ret(stocks).dropna(how="all").corr()
            heat = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1,
                colorscale=[[0, "#6687B0"], [0.5, "#F5F0EC"], [1, "#C47D5A"]],
                text=np.round(corr.values, 2), texttemplate="%{text}",
                hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>"))
            heat.update_layout(height=max(380, 60 * len(corr)),
                               margin=dict(l=55, r=30, t=30, b=55), yaxis=dict(autorange="reversed"))
            st.plotly_chart(heat, width="stretch", key="corr_heatmap")

hs = hist[[s for s in symbols if s in hist.columns]]
cap, mx = PROFILES[profile]

with tab_sig:
    st.subheader("Trend Signals & Suggested Allocation")
    st.warning("Educational model only, not investment advice. Signals are rule-based, "
               "unverified for future performance, and ignore fundamentals, taxes and your risk tolerance.")
    for k, v in DEFAULT_W.items():
        st.session_state.setdefault(k, v)
    with st.expander("Score weights"):
        wc = st.columns(4)
        wts = (wc[0].slider("Price vs MA200", 0, 100, key="w_trend"),
               wc[1].slider("MA50 vs MA200", 0, 100, key="w_cross"),
               wc[2].slider("6M momentum", 0, 100, key="w_mom"),
               wc[3].slider("RSI", 0, 100, key="w_rsi"))
        st.button("Reset weights", on_click=reset_weights)
        if sum(wts) == 0:
            st.warning("All weights are zero; every score will be 0.")
    ma200, mom, rsi, vol, score = indicators(hs, wts)
    last_price = hs.ffill().iloc[-1]
    tbl = pd.DataFrame({
        "Price": last_price,
        "vs MA200 %": (last_price / ma200.iloc[-1] - 1) * 100,
        "6M Momentum %": mom.iloc[-1] * 100,
        "RSI": rsi.iloc[-1],
        "Volatility %": vol.iloc[-1] * 100,
        "Score": score.iloc[-1],
    }).dropna(subset=["Score"])
    if tbl.empty:
        st.info("Not enough history (about 1 year of daily data) to compute signals.")
    else:
        tbl["Signal"] = np.select([tbl["Score"] >= thr, tbl["Score"] <= -thr], ["Bullish", "Bearish"], "Neutral")
        w = allocate(tbl["Score"], tbl["Volatility %"] / 100, cap, mx)
        tbl["Suggested %"] = w * 100
        tbl["Amount ($)"] = w * capital
        tbl["Shares"] = np.floor(tbl["Amount ($)"] / tbl["Price"])
        cash = 100 - tbl["Suggested %"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Suggested invested", f"{100 - cash:.1f}%")
        c2.metric("Suggested cash", f"{cash:.1f}%")
        c3.metric("Invested amount", f"${capital * (100 - cash) / 100:,.0f}")
        event = st.dataframe(
            tbl[["Price", "vs MA200 %", "6M Momentum %", "RSI", "Volatility %", "Score",
                 "Signal", "Suggested %", "Amount ($)", "Shares"]].style.format({
                "Price": "${:,.2f}", "vs MA200 %": "{:+.1f}%", "6M Momentum %": "{:+.1f}%",
                "RSI": "{:.0f}", "Volatility %": "{:.1f}%", "Score": "{:+.2f}",
                "Suggested %": "{:.1f}%", "Amount ($)": "${:,.0f}", "Shares": "{:,.0f}"}),
            width="stretch", on_select="rerun", selection_mode="single-row", key="sig_table")
        st.caption("Click a row to open it in the Stock Detail tab.")
        sel = tbl.index[event.selection.rows[0]] if event.selection.rows else None
        if sel is not None and sel != st.session_state.get("_last_sel"):
            st.session_state["detail"] = sel
        st.session_state["_last_sel"] = sel
        st.download_button("Download allocation CSV", tbl.to_csv().encode(), "allocation.csv", "text/csv")
        pos = tbl[tbl["Suggested %"] > 0]
        pie = go.Figure(go.Pie(
            labels=list(pos.index) + ["Cash"], values=list(pos["Suggested %"]) + [cash], hole=0.5,
            marker=dict(colors=COLORS[:len(pos)] + ["#D8DED8"])))
        pie.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(pie, width="stretch", key="alloc_pie")

with tab_det:
    opts = list(hs.columns)
    if st.session_state.get("detail") not in opts:
        st.session_state.pop("detail", None)
    t = st.selectbox("Stock", opts, key="detail")
    win = st.segmented_control("Window", list(DET_DAYS), default="1Y", key="det_win") or "1Y"
    s = hs[t].dropna()
    if len(s) < 60:
        st.info("Not enough history for this stock.")
    else:
        st.caption(names.get(t, ""))
        ma50 = s.rolling(50).mean()
        ma200_s = s.rolling(200).mean()
        rsi_s = calc_rsi(s)
        cut = s.index[-1] - pd.Timedelta(days=DET_DAYS[win])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last price", f"${s.iloc[-1]:,.2f}", f"{(s.iloc[-1] / s.iloc[-2] - 1) * 100:+.2f}%")
        c2.metric("RSI (14)", f"{rsi_s.iloc[-1]:.0f}")
        c3.metric("vs MA200", f"{(s.iloc[-1] / ma200_s.iloc[-1] - 1) * 100:+.1f}%"
                  if pd.notna(ma200_s.iloc[-1]) else "N/A")
        if t in tbl.index:
            c4.metric("Signal", tbl.loc[t, "Signal"], f"{tbl.loc[t, 'Score']:+.2f}")
        else:
            c4.metric("Signal", "N/A")
        det = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
        det.add_trace(go.Scatter(x=s.loc[cut:].index, y=s.loc[cut:], name=t, mode="lines",
                                 line=dict(color=COLORS[0], width=2.5)), row=1, col=1)
        det.add_trace(go.Scatter(x=ma50.loc[cut:].index, y=ma50.loc[cut:], name="MA50", mode="lines",
                                 line=dict(color=COLORS[1], width=1.5)), row=1, col=1)
        det.add_trace(go.Scatter(x=ma200_s.loc[cut:].index, y=ma200_s.loc[cut:], name="MA200", mode="lines",
                                 line=dict(color=COLORS[2], width=1.5)), row=1, col=1)
        det.add_trace(go.Scatter(x=rsi_s.loc[cut:].index, y=rsi_s.loc[cut:], name="RSI", mode="lines",
                                 line=dict(color=COLORS[3], width=1.5)), row=2, col=1)
        det.add_hline(y=70, line_dash="dot", row=2, col=1)
        det.add_hline(y=30, line_dash="dot", row=2, col=1)
        det.update_yaxes(title_text="Price (USD)", row=1, col=1)
        det.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        det.update_layout(height=620, hovermode="x unified", margin=dict(l=55, r=30, t=30, b=60),
                          legend=dict(orientation="h", y=-0.1, x=0))
        st.plotly_chart(det, width="stretch", key="detail_chart")

with tab_wi:
    st.subheader("What if I had invested?")
    wopts = list(hs.columns)
    c1, c2, c3 = st.columns(3)
    wt = c1.selectbox("Stock", wopts, key="wi_t")
    amount = c2.number_input("Amount (USD)", min_value=10.0, value=1000.0, step=100.0, key="wi_amt")
    dca = c3.radio("Strategy", ["Lump sum", "Monthly DCA"], horizontal=True, key="wi_mode") == "Monthly DCA"
    ws = hs[wt].dropna()
    min_d, max_d = ws.index[0].date(), ws.index[-1].date()
    cur = st.session_state.get("wi_date")
    if cur is not None and not (min_d <= cur <= max_d):
        del st.session_state["wi_date"]
    buy = st.date_input("Start date", value=min(max(date.today() - timedelta(days=365), min_d), max_d),
                        min_value=min_d, max_value=max_d, key="wi_date")
    px = ws.loc[pd.Timestamp(buy):]
    if len(px) < 2:
        st.info("Pick an earlier start date.")
    else:
        value, invested = simulate(px, amount, dca)
        gain = value.iloc[-1] - invested.iloc[-1]
        gain_pct = gain / invested.iloc[-1] * 100
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invested", f"${invested.iloc[-1]:,.0f}")
        m2.metric("Value today", f"${value.iloc[-1]:,.0f}", f"{gain_pct:+.1f}%")
        m3.metric("Gain / loss", f"{'+' if gain >= 0 else '-'}${abs(gain):,.0f}")
        if "SPY" in hist.columns and wt != "SPY":
            spx = hist["SPY"].dropna().loc[pd.Timestamp(buy):]
            if len(spx) >= 2:
                sv, si = simulate(spx, amount, dca)
                m4.metric("SPY, same plan", f"{(sv.iloc[-1] / si.iloc[-1] - 1) * 100:+.1f}%")
        wf = go.Figure()
        wf.add_trace(go.Scatter(x=value.index, y=value.values, mode="lines", name="Value",
                                line=dict(color=COLORS[0], width=2.5)))
        wf.add_trace(go.Scatter(x=invested.index, y=invested.values, mode="lines", name="Invested",
                                line=dict(color="#8A8F98", width=2, dash="dash", shape="hv")))
        style(wf, "USD")
        st.plotly_chart(wf, width="stretch", key="whatif_chart")
        st.caption("Uses adjusted closes, no fees or taxes. DCA buys on the first trading day of each month.")

with tab_bt:
    st.subheader("Monthly Rebalance Backtest")
    out = backtest(hs, cap, mx, rf, wts) if not hs.empty else None
    if out is None:
        st.info("Not enough history to run the backtest.")
    else:
        port, ew = out
        curves = {"Strategy": (1 + port.fillna(0)).cumprod() * 100,
                  "Equal-weight (monthly)": (1 + ew.fillna(0)).cumprod() * 100}
        if "SPY" in hist.columns:
            spy_r = ret(hist["SPY"]).reindex(port.index).fillna(0)
            curves["SPY"] = (1 + spy_r).cumprod() * 100
        bt = go.Figure()
        bt_colors = ["#6687B0", "#C47D5A", "#8A8F98"]
        for (name, cv), color in zip(curves.items(), bt_colors):
            bt.add_trace(go.Scatter(x=cv.index, y=cv.values, mode="lines", name=name,
                                    line=dict(color=color, width=2.5,
                                              dash="dash" if name == "SPY" else "solid")))
        style(bt, "Growth of 100")
        st.plotly_chart(bt, width="stretch", key="backtest_chart")
        bmdf = pd.DataFrame({k: perf(v, rf) for k, v in curves.items()}).T
        st.dataframe(bmdf.style.format(METRIC_FMT), width="stretch")
        st.caption("Signals use only data available at each month-end; weights apply from the next trading day. "
                   "No transaction costs or taxes; tickers are chosen today, so survivorship bias applies.")
