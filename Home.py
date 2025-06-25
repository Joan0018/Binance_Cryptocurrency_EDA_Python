import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import hmac
import hashlib
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Crypto EDA Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CLASS & FUNCTION DEFINITIONS
# Define all classes and functions here before they are called.
# ==============================================================================

# Binance API Wrapper Class
class BinanceAPI:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
    
    def _generate_signature(self, query_string):
        """Generate signature for authenticated requests"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint, params=None, signed=False):
        """Make request to Binance API"""
        if params is None:
            params = {}
        
        headers = {'X-MBX-APIKEY': self.api_key}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

# Binance API functions (and other helpers)
@st.cache_data(ttl=300)
def get_binance_symbols():
    """Get all available trading pairs from Binance"""
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
        return sorted(symbols)
    except Exception as e:
        st.error(f"Error fetching symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']

@st.cache_data(ttl=60)
def get_24hr_ticker(symbol):
    """Get 24hr ticker statistics"""
    try:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url)
        return response.json()
    except Exception as e:
        st.error(f"Error fetching ticker data: {e}")
        return None

@st.cache_data(ttl=300)
def get_kline_data(symbol, interval, limit=500):
    """Get historical kline/candlestick data"""
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 'number_of_trades']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'number_of_trades']]
    except Exception as e:
        st.error(f"Error fetching kline data: {e}")
        return None

def get_account_info(binance_api):
    """Get account information"""
    try:
        return binance_api._make_request("/api/v3/account", signed=True)
    except Exception as e:
        # We raise the exception here to be caught in the configuration block
        raise e

def get_open_orders(binance_api, symbol=None):
    """Get open orders"""
    try:
        params = {}
        if symbol:
            params['symbol'] = symbol
        return binance_api._make_request("/api/v3/openOrders", params, signed=True)
    except Exception as e:
        st.error(f"Error fetching open orders: {e}")
        return None

def get_order_history(binance_api, symbol, limit=500):
    """Get order history"""
    try:
        params = {'symbol': symbol, 'limit': limit}
        return binance_api._make_request("/api/v3/allOrders", params, signed=True)
    except Exception as e:
        st.error(f"Error fetching order history: {e}")
        return None

def get_trade_history(binance_api, symbol, limit=500):
    """Get trade history"""
    try:
        params = {'symbol': symbol, 'limit': limit}
        return binance_api._make_request("/api/v3/myTrades", params, signed=True)
    except Exception as e:
        st.error(f"Error fetching trade history: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    df = df.copy()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365)
    return df

# ==============================================================================
# APP LAYOUT & LOGIC
# ==============================================================================

# Initialize session state for API credentials
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; color: #FF6B35; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #1E1E1E; padding: 1rem; border-radius: 10px; border-left: 4px solid #FF6B35; }
    .stSelectbox > div > div { background-color: #2E2E2E; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">₿ Cryptocurrency EDA Dashboard</h1>', unsafe_allow_html=True)

# API Configuration Section
if not st.session_state.api_configured:
    st.warning("⚠️ Please configure your Binance API credentials to access advanced features")
    
    with st.expander("🔑 API Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password", help="Your Binance API Key")
        with col2:
            api_secret = st.text_input("API Secret", type="password", help="Your Binance API Secret")
        
        col1_btn, col2_btn, col3_btn = st.columns([1, 1, 2])
        with col1_btn:
            if st.button("🔧 Configure API"):
                if api_key and api_secret:
                    try:
                        binance_api = BinanceAPI(api_key, api_secret)
                        # Test the API connection - this will now work
                        account_info = get_account_info(binance_api)
                        
                        # Check for a specific key in the response to confirm success
                        if 'accountType' in account_info:
                            st.session_state.binance_api = binance_api
                            st.session_state.api_configured = True
                            st.success("✅ API configured successfully!")
                            st.rerun()
                        else:
                            # This handles cases where the API call succeeds but returns an error message
                            st.error(f"❌ API Error: {account_info.get('msg', 'Unknown error. Check credentials.')}")
                    except Exception as e:
                        st.error(f"❌ API Configuration Error: {str(e)}")
                else:
                    st.error("Please enter both API Key and API Secret")
        
        with col2_btn:
            if st.button("📊 Use Public Data Only"):
                st.session_state.api_configured = True
                st.session_state.public_only = True
                st.info("Using public data only. Some features will be limited.")
                st.rerun()
        
        st.info("""
        **API Permissions Required:**
        - ✅ Read Info (Spot & Margin Trading)
        - ❌ Enable Trading (Not required for EDA)
        - ❌ Enable Futures (Not required for EDA)
        
        **Security Note:** Your API credentials are only stored in this session and are not saved permanently.
        """)
    
    st.stop()

# --- The rest of your script remains the same ---
# Check if using authenticated API or public only
using_api = hasattr(st.session_state, 'binance_api') and not st.session_state.get('public_only', False)

# Sidebar
st.sidebar.title("📊 Analysis Settings")

# API Status
if using_api:
    st.sidebar.success("🔑 Authenticated API Active")
    if st.sidebar.button("🔄 Reset API"):
        for key in ['api_configured', 'binance_api', 'public_only']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
else:
    st.sidebar.info("📊 Public Data Mode")

# Analysis type selection
analysis_modes = ["Market Analysis", "Portfolio Analysis", "Trading Analysis"] if using_api else ["Market Analysis"]
selected_mode = st.sidebar.selectbox("Analysis Mode", analysis_modes)

# Get available symbols
symbols = get_binance_symbols()
selected_symbol = st.sidebar.selectbox("Select Trading Pair", symbols, index=symbols.index('BTCUSDT') if 'BTCUSDT' in symbols else 0)

# Time interval
intervals = {
    '1m': '1 minute', '5m': '5 minutes', '15m': '15 minutes', '30m': '30 minutes',
    '1h': '1 hour', '4h': '4 hours', '1d': '1 day', '1w': '1 week'
}
selected_interval = st.sidebar.selectbox("Select Time Interval", list(intervals.keys()), index=6)

# Data points
data_points = st.sidebar.slider("Number of Data Points", 50, 1000, 500)

# Fetch data
if st.sidebar.button("🔄 Refresh Data") or 'crypto_data' not in st.session_state:
    with st.spinner("Fetching data from Binance..."):
        # --- Standard Data Fetching ---
        ticker_data = get_24hr_ticker(selected_symbol)
        kline_data = get_kline_data(selected_symbol, selected_interval, data_points)
        
        # --- Enhanced Logic for Trading/Portfolio Analysis ---
        if using_api:
            try:
                # Get basic account info
                account_data = get_account_info(st.session_state.binance_api)
                st.session_state.account_data = account_data

                # For the selected symbol (for the "Trading Analysis" tab)
                st.session_state.open_orders = get_open_orders(st.session_state.binance_api, selected_symbol)
                st.session_state.order_history = get_order_history(st.session_state.binance_api, selected_symbol)
                st.session_state.trade_history = get_trade_history(st.session_state.binance_api, selected_symbol)

                # --- NEW: Fetch ALL trades for a comprehensive view ---
                st.info("Fetching all trade history... this may take a moment.")
                all_trades = []
                # Get symbols for assets you actually own
                if account_data and 'balances' in account_data:
                    assets = [b['asset'] for b in account_data['balances'] if float(b['free']) + float(b['locked']) > 0]
                    # Common quote assets to check against
                    quote_assets = ['USDT', 'BUSD', 'BTC', 'ETH', 'BNB']
                    
                    # Create a list of potential symbols to check
                    symbols_to_check = get_binance_symbols()
                    
                    # Filter for symbols relevant to the user's assets
                    relevant_symbols = [s for s in symbols_to_check if any(a + q == s for a in assets for q in quote_assets) or any(q + a == s for a in assets for q in quote_assets)]
                    
                    progress_bar = st.progress(0)
                    for i, symbol in enumerate(relevant_symbols):
                        trades = get_trade_history(st.session_state.binance_api, symbol)
                        if trades:
                            all_trades.extend(trades)
                        progress_bar.progress((i + 1) / len(relevant_symbols), text=f"Checking {symbol}")
                    
                    st.session_state.all_trade_history = all_trades
                    progress_bar.empty()

            except Exception as e:
                st.warning(f"Could not fetch all account data: {e}")
        
        if kline_data is not None:
            kline_data = calculate_technical_indicators(kline_data)
            st.session_state.crypto_data = kline_data
            st.session_state.ticker_data = ticker_data
            st.success("Data fetched successfully!")
        else:
            st.error("Failed to fetch data. Please try again.")
            st.stop()

# Main content
if 'crypto_data' in st.session_state:
    df = st.session_state.crypto_data
    ticker = st.session_state.ticker_data
    
    # Analysis Mode Routing
    if selected_mode == "Market Analysis":
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Price Analysis", "📈 Technical Indicators", "📉 Volume Analysis", "🔍 Statistical Summary", "📋 Raw Data"])
        
        with tab1:
            st.subheader("Price Movement Analysis")
            
            # Candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=('Price Action', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color='rgba(158, 71, 99, 0.6)'), row=2, col=1)
            
            fig.update_layout(title=f'{selected_symbol} Price Analysis', xaxis_rangeslider_visible=False, height=600, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            # Price distribution
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df, x='close', nbins=50, title='Price Distribution', template='plotly_dark')
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_returns = px.histogram(df, x='returns', nbins=50, title='Returns Distribution', template='plotly_dark')
                st.plotly_chart(fig_returns, use_container_width=True)
        
        with tab2:
            st.subheader("Technical Indicators")
            
            # RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
            fig_rsi.update_layout(title='RSI (Relative Strength Index)', template='plotly_dark')
            st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')), row=1, col=1)
            fig_macd.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='red')), row=1, col=1)
            fig_macd.add_trace(go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram'), row=2, col=1)
            fig_macd.update_layout(title='MACD', template='plotly_dark')
            st.plotly_chart(fig_macd, use_container_width=True)
            
            # Bollinger Bands
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='white')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='Upper Band', line=dict(color='red')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='Lower Band', line=dict(color='green')))
            fig_bb.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name='Middle Band', line=dict(color='blue')))
            fig_bb.update_layout(title='Bollinger Bands', template='plotly_dark')
            st.plotly_chart(fig_bb, use_container_width=True)
        
        with tab3:
            st.subheader("Volume Analysis")
            col1, col2 = st.columns(2)
            with col1:
                fig_vol = px.bar(df.reset_index(), x='timestamp', y='volume', title='Volume Over Time', template='plotly_dark')
                st.plotly_chart(fig_vol, use_container_width=True)
            with col2:
                fig_corr = px.scatter(df, x='volume', y='close', title='Price vs Volume Correlation', template='plotly_dark')
                st.plotly_chart(fig_corr, use_container_width=True)
            
            fig_vol_dist = px.histogram(df, x='volume', nbins=50, title='Volume Distribution', template='plotly_dark')
            st.plotly_chart(fig_vol_dist, use_container_width=True)
        
        with tab4:
            st.subheader("Statistical Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Price Statistics**")
                st.dataframe(df[['open', 'high', 'low', 'close']].describe())
                st.write("**Volatility Metrics**")
                volatility_stats = {
                    'Current Volatility': f"{df['volatility'].iloc[-1]:.4f}",
                    'Average Volatility': f"{df['volatility'].mean():.4f}",
                    'Max Volatility': f"{df['volatility'].max():.4f}",
                    'Min Volatility': f"{df['volatility'].min():.4f}"
                }
                for key, value in volatility_stats.items():
                    st.metric(key, value)
            with col2:
                st.write("**Volume Statistics**")
                st.dataframe(df[['volume', 'number_of_trades']].describe())
                st.write("**Technical Indicators Summary**")
                tech_stats = {
                    'Current RSI': f"{df['rsi'].iloc[-1]:.2f}",
                    'Current MACD': f"{df['macd'].iloc[-1]:.6f}",
                    'Price vs SMA20': f"{((df['close'].iloc[-1] / df['sma_20'].iloc[-1] - 1) * 100):.2f}%",
                    'Price vs SMA50': f"{((df['close'].iloc[-1] / df['sma_50'].iloc[-1] - 1) * 100):.2f}%"
                }
                for key, value in tech_stats.items():
                    st.metric(key, value)
            
            st.write("**Correlation Matrix**")
            corr_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'volatility']
            corr_matrix = df[corr_columns].corr()
            fig_corr_matrix = px.imshow(corr_matrix, title='Feature Correlation Matrix', template='plotly_dark', color_continuous_scale='RdBu')
            st.plotly_chart(fig_corr_matrix, use_container_width=True)
        
        with tab5:
            st.subheader("Raw Data")
            col1, col2 = st.columns(2)
            with col1:
                show_indicators = st.checkbox("Show Technical Indicators", value=False)
            with col2:
                rows_to_show = st.slider("Rows to display", 10, len(df), 50)
            
            display_df = df if show_indicators else df[['open', 'high', 'low', 'close', 'volume', 'number_of_trades']]
            st.dataframe(display_df.head(rows_to_show))
            
            csv = display_df.to_csv()
            st.download_button(label="📥 Download Data as CSV", data=csv, file_name=f'{selected_symbol}_{selected_interval}_data.csv', mime='text/csv')
    
    elif selected_mode == "Portfolio Analysis" and using_api:
        st.subheader("💼 Portfolio Overview")
        if 'account_data' in st.session_state and st.session_state.account_data:
            account = st.session_state.account_data
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Account Type", account.get('accountType', 'N/A'))
            with col2: st.metric("Can Trade", "✅" if account.get('canTrade', False) else "❌")
            with col3: st.metric("Can Withdraw", "✅" if account.get('canWithdraw', False) else "❌")
            balances = [b for b in account.get('balances', []) if float(b['free']) + float(b['locked']) > 0]
            if balances:
                st.subheader("💰 Current Balances")
                balance_df = pd.DataFrame(balances)
                balance_df['free'] = pd.to_numeric(balance_df['free'])
                balance_df['locked'] = pd.to_numeric(balance_df['locked'])
                balance_df['total'] = balance_df['free'] + balance_df['locked']
                balance_df = balance_df[balance_df['total'] > 0].sort_values('total', ascending=False)
                fig_portfolio = px.pie(balance_df.head(10), values='total', names='asset', title='Portfolio Distribution (Top 10)', template='plotly_dark')
                st.plotly_chart(fig_portfolio, use_container_width=True)
                st.dataframe(balance_df[['asset', 'free', 'locked', 'total']], use_container_width=True)
            else: st.info("No balances found in your account.")
        
    elif selected_mode == "Trading Analysis" and using_api:
        st.subheader("📈 Trading Analysis")
        tab1, tab2, tab3 = st.tabs(["🔄 Open Orders", "📊 Order History", "💹 Trade History"])
        
        with tab1:
            if 'open_orders' in st.session_state and st.session_state.open_orders:
                orders_df = pd.DataFrame(st.session_state.open_orders)
                if not orders_df.empty:
                    orders_df['time'] = pd.to_datetime(pd.to_numeric(orders_df['time']), unit='ms')
                    st.dataframe(orders_df, use_container_width=True)
                else: st.info(f"No open orders found for {selected_symbol}.")
            else: st.info("No open orders data available.")
        
        with tab2:
            if 'order_history' in st.session_state and st.session_state.order_history:
                history_df = pd.DataFrame(st.session_state.order_history)
                if not history_df.empty:
                    history_df['time'] = pd.to_datetime(pd.to_numeric(history_df['time']), unit='ms')
                    fig_status = px.pie(history_df, names='status', title='Order Status Distribution', template='plotly_dark')
                    st.plotly_chart(fig_status, use_container_width=True)
                    daily_orders = history_df.groupby(history_df['time'].dt.date).size().reset_index(name='count')
                    fig_orders = px.bar(daily_orders, x='date', y='count', title='Orders per Day', template='plotly_dark')
                    st.plotly_chart(fig_orders, use_container_width=True)
                    st.dataframe(history_df, use_container_width=True)
                else: st.info(f"No order history found for {selected_symbol}.")
            else: st.info("No order history data available.")
        
        with tab3:
            if 'trade_history' in st.session_state and st.session_state.trade_history:
                trades_df = pd.DataFrame(st.session_state.trade_history)
                if not trades_df.empty:
                    trades_df['time'] = pd.to_datetime(pd.to_numeric(trades_df['time']), unit='ms')
                    trades_df['price'] = pd.to_numeric(trades_df['price'])
                    trades_df['qty'] = pd.to_numeric(trades_df['qty'])
                    trades_df['quoteQty'] = pd.to_numeric(trades_df['quoteQty'])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Total Trades", len(trades_df))
                    with col2: st.metric("Total Volume", f"{trades_df['qty'].sum():.4f}")
                    with col3: st.metric("Total Value", f"${trades_df['quoteQty'].sum():.2f}")
                    with col4:
                        avg_price = trades_df['quoteQty'].sum() / trades_df['qty'].sum() if trades_df['qty'].sum() > 0 else 0
                        st.metric("Avg Price", f"${avg_price:.4f}")
                    fig_trades = px.scatter(trades_df, x='time', y='price', size='qty', color='isBuyer', title='Trade History', template='plotly_dark')
                    st.plotly_chart(fig_trades, use_container_width=True)
                    st.dataframe(trades_df, use_container_width=True)
                else: st.info(f"No trade history found for {selected_symbol}.")
            else: st.info("No trade history data available.")

else:
    st.warning("Please fetch data using the refresh button in the sidebar.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'><p>Binance API | <a href='https://binance-docs.github.io/apidocs/spot/en/' target='_blank'>API Documentation</a></p></div>", unsafe_allow_html=True)