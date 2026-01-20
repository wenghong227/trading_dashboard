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
import traceback

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="ProQuant Analytics Suite",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv(override=True)

# INSTITUTIONAL CSS
st.markdown("""
<style>
    .stApp { background-color: #000000; }
    .metric-container { background-color: #111; border: 1px solid #333; padding: 15px; border-radius: 6px; }
    .metric-label { color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; }
    .metric-value { color: #fff; font-size: 22px; font-family: 'SF Mono', 'Roboto Mono', monospace; font-weight: 600; }
    .pos-val { color: #00e676; }
    .neg-val { color: #ff1744; }
    [data-testid="stDataFrame"] { border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. CONNECTION LAYER
# ---------------------------------------------------------
@st.cache_resource
def get_exchange(mode):
    try:
        is_live = mode == "Live Execution"
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
# 3. ADVANCED DATA FETCHING (PAGINATION ENGINE)
# ---------------------------------------------------------

def fetch_ohlcv_pagination(exchange, symbol, timeframe, since_ts):
    """
    Smart Pagination: Loops through API until it reaches 'Now'.
    Fixes the 'cutoff at Dec 31st' issue.
    """
    all_candles = []
    current_since = since_ts
    # Safety: Stop looking if we go past "Now"
    now_ts = int(datetime.now().timestamp() * 1000)
    
    while True:
        try:
            # Fetch batch (Limit 1000 is safer max for Bitget)
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            
            if not candles:
                break
            
            all_candles.extend(candles)
            
            # Get timestamp of the last candle found
            last_candle_ts = candles[-1][0]
            
            # If the last candle is close to "now", we are done
            if last_candle_ts >= (now_ts - 60000 * 240): # Within last 4 hours
                break
                
            # If we didn't get a full batch, we probably exhausted history
            if len(candles) < 2: 
                break
                
            # Move the pointer: Start next batch 1ms after the last candle
            current_since = last_candle_ts + 1
            
            # Rate limit protection (tiny sleep if not handled by ccxt)
            time.sleep(0.1) 
            
        except Exception as e:
            # If error occurs, return what we have so far
            break
            
    return all_candles

@st.cache_data(ttl=60)
def fetch_analytics_data(_exchange, symbol, days=30):
    cols_trades = ['Time', 'Side', 'Price', 'Amount', 'Pnl', 'Fee', 'OrderId', 'Position_Side']
    cols_daily = ['Date', 'Daily_PnL']
    
    empty_trades = pd.DataFrame(columns=cols_trades)
    empty_daily = pd.DataFrame(columns=cols_daily)
    empty_ohlcv = pd.DataFrame(columns=['Time', 'Open', 'High', 'Low', 'Close'])
    
    if not _exchange: return empty_ohlcv, empty_trades, empty_daily, 0.0, "No Connection"

    try:
        # Calculate 'Since' Timestamp
        since_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # 1. MARKET DATA (Using Pagination Fix)
        candles = fetch_ohlcv_pagination(_exchange, symbol, '4h', since_ts)
        
        if candles:
            df_ohlcv = pd.DataFrame(candles, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Vol'])
            df_ohlcv['Time'] = pd.to_datetime(df_ohlcv['Time'], unit='ms')
        else:
            df_ohlcv = empty_ohlcv
        
        # 2. TRADES (Usually Bitget returns enough trades, but we use 'since' here too)
        trades = _exchange.fetch_my_trades(symbol, since=since_ts, limit=100, params={'productType': 'umcbl'})
        if not trades: 
            return df_ohlcv, empty_trades, empty_daily, 0.0, None

        # 3. PROCESS TRADES
        t_data = []
        for t in trades:
            raw = t.get('info', {})
            pnl = float(t.get('realizedPnl') or raw.get('fillPnL') or raw.get('profit') or 0)
            
            t_data.append({
                'Time': datetime.fromtimestamp(t['timestamp']/1000),
                'Side': t['side'].upper(), 
                'Price': float(t['price']),
                'Amount': float(t['amount']),
                'Pnl': pnl,
                'Fee': float(t['fee']['cost'] if t.get('fee') else 0),
                'OrderId': t['order']
            })
        
        df_raw = pd.DataFrame(t_data)
        
        # 4. AGGREGATE SPLIT TRANSACTIONS
        if not df_raw.empty:
            df_raw['Is_Close'] = df_raw['Pnl'] != 0
            
            agg_rules = {
                'Time': 'first',
                'Side': 'first',
                'Price': 'mean', 
                'Amount': 'sum',
                'Pnl': 'sum',
                'Fee': 'sum'
            }
            if 'OrderId' in df_raw.columns:
                 df_agg = df_raw.groupby('OrderId', as_index=False).agg(agg_rules)
            else:
                 df_raw['Time_Group'] = df_raw['Time'].dt.round('1s')
                 df_agg = df_raw.groupby(['Time_Group', 'Side'], as_index=False).agg(agg_rules)

            df_agg = df_agg.sort_values("Time", ascending=False)
        else:
            df_agg = empty_trades

        # 5. POSITION SIDE LOGIC
        if not df_agg.empty and 'Pnl' in df_agg.columns:
            def determine_side(row):
                if row['Pnl'] != 0: 
                    if row['Side'] == 'BUY': return 'SHORT'
                    if row['Side'] == 'SELL': return 'LONG'
                return '-'
            df_agg['Position_Side'] = df_agg.apply(determine_side, axis=1)
        else:
            df_agg = empty_trades

        # 6. FILTER CLOSED
        if 'Pnl' in df_agg.columns:
            df_closed = df_agg[df_agg['Pnl'] != 0].copy()
        else:
            df_closed = empty_trades
        
        if df_closed.empty:
            return df_ohlcv, df_agg, empty_daily, 0.0, None

        # 7. DAILY AGGREGATION
        df_closed['Date'] = df_closed['Time'].dt.date
        df_daily = df_closed.groupby('Date')['Pnl'].sum().reset_index()
        df_daily.columns = ['Date', 'Daily_PnL']
        
        return df_ohlcv, df_agg, df_daily, 0.0, None

    except Exception as e:
        return empty_ohlcv, empty_trades, empty_daily, 0.0, str(e)

def calculate_advanced_metrics(df_closed):
    if df_closed.empty: return {}
    
    pnl = df_closed['Pnl']
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    
    total_trades = len(df_closed)
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0
    
    std_dev = pnl.std()
    sharpe = (pnl.mean() / std_dev) if std_dev != 0 else 0

    return {
        "Total PnL": pnl.sum(),
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Sharpe": sharpe,
        "Count": total_trades
    }

# ---------------------------------------------------------
# 4. UI HELPERS (FIXED FORMATTING)
# ---------------------------------------------------------
def kpi_card(col, label, value, is_currency=True, color=False):
    # 1. Handle Currency
    if is_currency:
        fmt = f"${value:,.2f}"
    else:
        # 2. Handle Integers (Trades, Counts) - No decimals
        if label in ["Total Trades", "Count"]:
            fmt = f"{int(value)}"
        # 3. Handle Ratios (Sharpe, Profit Factor) - 2 decimals, no %
        elif label in ["Profit Factor", "Sharpe", "Sortino", "Sharpe Ratio"]:
            fmt = f"{value:.2f}"
        # 4. Handle Percentages (Win Rate, ROI)
        else:
            fmt = f"{value:.2f}%"
    
    cls = ""
    if color: cls = "pos-val" if value >= 0 else "neg-val"
        
    col.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{fmt}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DASHBOARD LAYOUT
# ---------------------------------------------------------
st.sidebar.title("💠 ProQuant")
mode = st.sidebar.radio("MODE", ("Demo Simulation", "Live Execution"))
exchange = get_exchange(mode)

st.sidebar.markdown("---")
st.sidebar.caption("STRATEGY CONFIG")
symbol = st.sidebar.selectbox("Asset", ["ETH/USDT:USDT", "BTC/USDT:USDT", "SHIB/USDT:USDT"])
days_back = st.sidebar.slider("Lookback (Days)", 7, 90, 30)

if st.sidebar.button("RUN DIAGNOSTICS", type="primary", use_container_width=True):
    st.cache_data.clear()

if not exchange:
    st.error("🔌 Connection Failed.")
else:
    df_ohlcv, df_raw, df_daily, eq, err = fetch_analytics_data(exchange, symbol, days_back)
    
    df_closed = pd.DataFrame()
    if not df_raw.empty and 'Pnl' in df_raw.columns:
        df_closed = df_raw[df_raw['Pnl'] != 0].copy()

    st.markdown(f"## 📊 Strategy Performance: {symbol.split('/')[0]}")
    
    # --- DEBUG SECTION FOR DATA ---
    # st.write(f"Candles Fetched: {len(df_ohlcv)}") # Uncomment to check candle count
    # if not df_ohlcv.empty:
    #    st.write(f"Date Range: {df_ohlcv['Time'].min()} to {df_ohlcv['Time'].max()}")

    if df_closed.empty:
        st.info(f"No closed trade data available for {symbol} in the last {days_back} days.")
        if err: st.error(f"Debug Info: {err}")
    else:
        metrics = calculate_advanced_metrics(df_closed)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi_card(c1, "Net Profit", metrics['Total PnL'], color=True)
        kpi_card(c2, "Win Rate", metrics['Win Rate'], is_currency=False, color=True)
        kpi_card(c3, "Profit Factor", metrics['Profit Factor'], is_currency=False)
        kpi_card(c4, "Sharpe Ratio", metrics['Sharpe'], is_currency=False)
        kpi_card(c5, "Total Trades", metrics['Count'], is_currency=False)

        st.markdown("---")

        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            df_curve = df_closed.sort_values("Time").copy()
            df_curve['Strategy_Equity'] = df_curve['Pnl'].cumsum()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Strategy
            fig.add_trace(go.Scatter(x=df_curve['Time'], y=df_curve['Strategy_Equity'], name="Strategy PnL", line=dict(color='#00e676', width=2), fill='tozeroy'), secondary_y=False)
            
            # Asset Price (Now fetches full history)
            if not df_ohlcv.empty:
                fig.add_trace(go.Scatter(x=df_ohlcv['Time'], y=df_ohlcv['Close'], name="Asset Price", line=dict(color='#ffffff', width=1, dash='dot')), secondary_y=True)

            fig.update_layout(title="Alpha Generation", plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#888', height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_side:
            fig_hist = px.histogram(df_closed, x="Pnl", nbins=15, title="PnL Distribution", color_discrete_sequence=['#58a6ff'])
            fig_hist.update_layout(plot_bgcolor='#111', paper_bgcolor='#111', font_color='#888', height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

        t1, t2, t3 = st.tabs(["🔥 Heatmap", "⚖️ Long vs Short", "📜 Aggregated Ledger"])

        with t1:
            if not df_daily.empty:
                if 'Date' in df_daily.columns and 'Daily_PnL' in df_daily.columns:
                    fig_cal = px.bar(df_daily, x='Date', y='Daily_PnL', title="Daily Performance", color='Daily_PnL', color_continuous_scale=['#ff1744', '#111', '#00e676'])
                    fig_cal.update_layout(plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', font_color='#888')
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.warning("Insufficient daily data.")

        with t2:
            if 'Position_Side' in df_closed.columns:
                longs = df_closed[df_closed['Position_Side'] == 'LONG']
                shorts = df_closed[df_closed['Position_Side'] == 'SHORT']
                
                c_l, c_s = st.columns(2)
                with c_l:
                    st.markdown("#### Long Performance")
                    if not longs.empty: st.dataframe(pd.DataFrame([calculate_advanced_metrics(longs)]).T, use_container_width=True)
                    else: st.caption("No Longs Closed")
                with c_s:
                    st.markdown("#### Short Performance")
                    if not shorts.empty: st.dataframe(pd.DataFrame([calculate_advanced_metrics(shorts)]).T, use_container_width=True)
                    else: st.caption("No Shorts Closed")
            else:
                st.warning("Position side data unavailable.")

        with t3:
            cols_to_show = ['Time', 'Position_Side', 'Price', 'Amount', 'Pnl', 'Fee']
            available_cols = [c for c in cols_to_show if c in df_closed.columns]
            
            grid_df = df_closed[available_cols].copy()
            st.dataframe(
                grid_df.sort_values("Time", ascending=False),
                use_container_width=True,
                column_config={
                    "Time": st.column_config.DatetimeColumn("Date", format="D MMM HH:mm"),
                    "Pnl": st.column_config.NumberColumn("Net PnL", format="$%.2f"),
                    "Position_Side": st.column_config.TextColumn("Position"),
                    "Amount": st.column_config.NumberColumn("Size"),
                }
            )

    if err: st.error(err)