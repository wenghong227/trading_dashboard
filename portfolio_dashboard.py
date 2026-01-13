import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# ---------------------------------------------------------
# 1. CONFIGURATION & THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="ProTrader X Terminal",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(override=True)

# ADVANCED CSS
st.markdown("""
<style>
    /* Dark Theme Base */
    .stApp { background-color: #0e1117; }
    
    /* Metrics Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: #58a6ff; transform: translateY(-2px); }
    
    /* Typography */
    .metric-label { color: #8b949e; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; }
    .metric-value { color: #f0f6fc; font-size: 26px; font-family: 'SF Mono', 'Roboto Mono', monospace; font-weight: 700; margin-top: 5px; }
    
    /* Profit/Loss Colors */
    .pnl-pos { color: #3fb950; font-weight: bold; }
    .pnl-neg { color: #f85149; font-weight: bold; }
    
    /* Custom Dividers */
    hr { margin: 2em 0; border: 0; border-top: 1px solid #30363d; }
    
    /* Pulse Animation for Live Status */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(63, 185, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
    }
    .live-indicator {
        width: 10px; height: 10px; background-color: #3fb950; border-radius: 50%;
        display: inline-block; margin-right: 8px; animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. SYSTEM ARCHITECTURE
# ---------------------------------------------------------
@st.cache_resource
def get_exchange(mode):
    try:
        is_live = mode == "Live Trading"
        prefix = "BITGET_" if is_live else "BITGET_DEMO_"
        
        exchange = ccxt.bitget({
            'apiKey': os.getenv(f"{prefix}API_KEY"),
            'secret': os.getenv(f"{prefix}API_SECRET"),
            'password': os.getenv(f"{prefix}API_PASSWORD"),
            'enableRateLimit': True,
            'options': {'defaultType': 'swap', 'adjustForTimeDifference': True}
        })
        exchange.set_sandbox_mode(not is_live)
        return exchange
    except: return None

# ---------------------------------------------------------
# 3. QUANTITATIVE ENGINE
# ---------------------------------------------------------
def calculate_indicators(df):
    """Adds Strategy Indicators (RSI, BB) for Visualization"""
    if df.empty: return df
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['SMA20'] + (df['STD20'] * 2)
    df['LowerBB'] = df['SMA20'] - (df['STD20'] * 2)
    
    return df

@st.cache_data(ttl=10)
def fetch_market_pulse(mode):
    """Fetches current prices for the Watchlist Bar"""
    if not exchange: return {}
    tickers = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
    data = {}
    try:
        raw = exchange.fetch_tickers(tickers)
        for t in tickers:
            if t in raw:
                data[t] = raw[t]['last']
    except: pass
    return data

@st.cache_data(ttl=60)
def fetch_full_data(mode, symbol, days_back=30):
    """Master Data Fetcher: Balance, OHLCV, Trades"""
    if not exchange: return {}, pd.DataFrame(), pd.DataFrame()
    
    # 1. BALANCE
    balance = {'equity': 0.0, 'available': 0.0}
    try:
        bal = exchange.fetch_balance()
        if 'info' in bal and isinstance(bal['info'], list):
            for a in bal['info']:
                if a.get('marginCoin')=='USDT':
                    balance['equity'] = float(a.get('usdtEquity',0))
                    balance['available'] = float(a.get('available',0))
                    break
        else:
            balance['equity'] = bal.get('total', {}).get('USDT', 0)
            balance['available'] = bal.get('free', {}).get('USDT', 0)
    except: pass

    # 2. OHLCV (Market Data)
    df_ohlcv = pd.DataFrame()
    try:
        # Fetch enough data for indicators
        candles = exchange.fetch_ohlcv(symbol, timeframe='4h', limit=100)
        if candles:
            df_ohlcv = pd.DataFrame(candles, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_ohlcv['Time'] = pd.to_datetime(df_ohlcv['Time'], unit='ms')
            df_ohlcv = calculate_indicators(df_ohlcv)
    except: pass

    # 3. TRADES
    df_trades = pd.DataFrame()
    try:
        # Look back further to ensure we catch trades
        since_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        trades = exchange.fetch_my_trades(symbol, since=since_ts, limit=100, params={'productType': 'umcbl'})
        
        trade_data = []
        for t in trades:
            raw = t.get('info', {})
            pnl = float(t.get('realizedPnl') or raw.get('fillPnL') or raw.get('profit') or 0)
            trade_data.append({
                'Time': datetime.fromtimestamp(t['timestamp'] / 1000),
                'Side': t['side'].upper(),
                'Price': t['price'],
                'Size': t['amount'],
                'Fee': t['fee']['cost'] if t.get('fee') else 0,
                'RealizedPnl': pnl,
                'Status': 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'OPEN')
            })
        df_trades = pd.DataFrame(trade_data)
    except: pass

    return balance, df_ohlcv, df_trades

# ---------------------------------------------------------
# 4. FRONTEND UI
# ---------------------------------------------------------
st.sidebar.title("💠 ProTrader X")
mode = st.sidebar.radio("ENVIRONMENT", ("Demo Trading", "Live Trading"))
exchange = get_exchange(mode)

# Sidebar Controls
st.sidebar.markdown("---")
st.sidebar.caption("STRATEGY SETTINGS")
symbol = st.sidebar.selectbox("Active Asset", ["ETH/USDT:USDT", "BTC/USDT:USDT", "SOL/USDT:USDT"])
history_window = st.sidebar.slider("History Lookback (Days)", 7, 90, 30)

if st.sidebar.button("⚡ REFRESH SYSTEM", type="primary", use_container_width=True):
    st.cache_data.clear()

if not exchange:
    st.error("🔌 SYSTEM DISCONNECTED: Check API Keys")
else:
    # --- A. WATCHLIST BAR ---
    pulse_data = fetch_market_pulse(mode)
    if pulse_data:
        cols = st.columns(len(pulse_data) + 1)
        cols[0].markdown(f"#### <span class='live-indicator'></span> {mode.split()[0].upper()}", unsafe_allow_html=True)
        for idx, (ticker, price) in enumerate(pulse_data.items()):
            cols[idx+1].metric(ticker.split('/')[0], f"${price:,.2f}")
    st.markdown("---")

    # --- B. FETCH MAIN DATA ---
    bal, df_ohlcv, df_trades = fetch_full_data(mode, symbol, history_window)
    
    # Metrics Logic
    df_closed = df_trades[df_trades['RealizedPnl'] != 0] if not df_trades.empty else pd.DataFrame()
    
    total_pnl = df_closed['RealizedPnl'].sum() if not df_closed.empty else 0
    win_rate = 0
    if not df_closed.empty:
        wins = len(df_closed[df_closed['RealizedPnl'] > 0])
        win_rate = (wins / len(df_closed)) * 100

    # --- C. METRIC CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    
    def card(col, label, value, delta=None, is_currency=True):
        fmt = f"${value:,.2f}" if is_currency else f"{value:.1f}%"
        delta_html = ""
        if delta is not None:
            color = "#3fb950" if delta >= 0 else "#f85149"
            sign = "+" if delta >= 0 else ""
            delta_html = f"<span style='color:{color}; font-size:14px; font-weight:bold;'>{sign}{delta:,.2f}</span>"
        
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{fmt}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)

    card(c1, "Total Equity", bal['equity'])
    card(c2, "Available Margin", bal['available'])
    card(c3, "Net PnL (Period)", total_pnl, delta=total_pnl)
    card(c4, "Win Rate", win_rate, is_currency=False)

    # --- D. STRATEGY INSPECTOR (CHARTS) ---
    st.markdown("### 📊 Strategy Inspector")
    
    tab1, tab2 = st.tabs(["Technical Analysis", "Performance Curve"])
    
    with tab1:
        if not df_ohlcv.empty:
            # Create Subplots: Row 1 = Price + BB, Row 2 = RSI
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=[0.7, 0.3])

            # 1. Candlestick
            fig.add_trace(go.Candlestick(
                x=df_ohlcv['Time'], open=df_ohlcv['Open'], high=df_ohlcv['High'],
                low=df_ohlcv['Low'], close=df_ohlcv['Close'], name='Price'
            ), row=1, col=1)

            # 2. Bollinger Bands
            fig.add_trace(go.Scatter(x=df_ohlcv['Time'], y=df_ohlcv['UpperBB'], 
                                     line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='Upper BB'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_ohlcv['Time'], y=df_ohlcv['LowerBB'], 
                                     line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='Lower BB', fill='tonexty'), row=1, col=1)

            # 3. Trade Markers
            if not df_trades.empty:
                entries = df_trades[df_trades['RealizedPnl'] == 0]
                exits = df_trades[df_trades['RealizedPnl'] != 0]
                
                # Entries
                fig.add_trace(go.Scatter(x=entries['Time'], y=entries['Price'], mode='markers', 
                                         marker=dict(symbol='triangle-up', size=12, color='#2196F3'), name='Entry'), row=1, col=1)
                # Exits
                fig.add_trace(go.Scatter(x=exits['Time'], y=exits['Price'], mode='markers', 
                                         marker=dict(symbol='triangle-down', size=12, color=exits['RealizedPnl'].apply(lambda x: '#00e676' if x>0 else '#ff1744')), name='Exit'), row=1, col=1)

            # 4. RSI
            fig.add_trace(go.Scatter(x=df_ohlcv['Time'], y=df_ohlcv['RSI'], line=dict(color='#ab47bc', width=2), name='RSI'), row=2, col=1)
            # RSI Lines
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)

            fig.update_layout(
                height=600, 
                plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font_color='#8b949e',
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor='#1f2937')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for market data...")

    with tab2:
        if not df_closed.empty:
            df_closed = df_closed.sort_values("Time")
            df_closed['Cumulative'] = df_closed['RealizedPnl'].cumsum()
            fig_area = px.area(df_closed, x='Time', y='Cumulative', title="Equity Growth", color_discrete_sequence=['#58a6ff'])
            fig_area.update_layout(plot_bgcolor='#0e1117', paper_bgcolor='#0e1117', font_color='#8b949e')
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.caption("No closed trades to display equity curve.")

    # --- E. INSTITUTIONAL LEDGER ---
    st.markdown("### 📜 Institutional Ledger")
    
    if not df_trades.empty:
        # Prepare table for display
        df_display = df_trades.copy()
        df_display = df_display.sort_values("Time", ascending=False)
        
        # Configure columns for slick visual
        st.dataframe(
            df_display,
            column_config={
                "Time": st.column_config.DatetimeColumn("Timestamp", format="D MMM HH:mm"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Size": st.column_config.NumberColumn("Size", format="%.4f"),
                "RealizedPnl": st.column_config.ProgressColumn(
                    "Net PnL",
                    help="Profit/Loss",
                    format="$%.2f",
                    min_value=-50, max_value=50, # Adjustable based on your typical trade size
                ),
                "Status": st.column_config.TextColumn("State"),
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Download Button
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Trade Log (CSV)",
            data=csv,
            file_name=f'trade_log_{symbol.split("/")[0]}.csv',
            mime='text/csv',
        )
    else:
        st.info("No trades found in the selected period.")