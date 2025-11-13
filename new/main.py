# ---------------------------
# Enhanced Cryptocurrency Analysis & Prediction System
# ---------------------------

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Crypto Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title and Introduction
# ---------------------------
st.markdown('<h1 class="main-header">🚀 Cryptocurrency Intelligence Platform</h1>', unsafe_allow_html=True)
st.markdown("""
    **Advanced AI-Powered Crypto Analysis System** | Real-time Data • Predictive Analytics • Asset Discovery • Risk Assessment
""")

# ---------------------------
# MODULE 1: DATA ACQUISITION
# ---------------------------
@st.cache_data(ttl=3600)
def load_crypto_data(coin, period):
    """Module 1: Data Acquisition - Loads cryptocurrency data from Yahoo Finance."""
    try:
        df = yf.download(coin, period=period, interval="1d", auto_adjust=True, progress=False)
        if df.empty:
            return None
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_multiple_cryptos(coins, period):
    """Load multiple cryptocurrencies for comparative analysis."""
    data_dict = {}
    for coin in coins:
        df = load_crypto_data(coin, period)
        if df is not None:
            data_dict[coin] = df
    return data_dict

# ---------------------------
# MODULE 2: DATA PRE-PROCESSING
# ---------------------------
def preprocess_data(df):
    """Module 2: Data Pre-processing - Clean and prepare data."""
    df = df.copy()
    
    # Missing value treatment
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Outlier detection using IQR
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df['Close'] < lower_bound) | (df['Close'] > upper_bound)).sum()
    
    # Cap outliers instead of removing them (to preserve data)
    df.loc[df['Close'] < lower_bound, 'Close'] = lower_bound
    df.loc[df['Close'] > upper_bound, 'Close'] = upper_bound
    
    return df, outliers

# ---------------------------
# MODULE 3: FEATURE ENGINEERING
# ---------------------------
def create_technical_indicators(df):
    """Create comprehensive technical indicators for analysis."""
    df = df.copy()
    
    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    
    # Volatility measures
    df['Volatility_7'] = df['Return'].rolling(window=7).std()
    df['Volatility_21'] = df['Return'].rolling(window=21).std()
    df['Volatility_50'] = df['Return'].rolling(window=50).std()
    
    # Price momentum
    df['Momentum_7'] = df['Close'] - df['Close'].shift(7)
    df['Momentum_21'] = df['Close'] - df['Close'].shift(21)
    
    # Volume indicators
    df['Volume_SMA_7'] = df['Volume'].rolling(window=7).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_7']
    
    # Price range
    df['Daily_Range'] = df['High'] - df['Low']
    df['Price_Change'] = df['Close'] - df['Open']
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_%D'] = df['Stochastic_%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    df = df.dropna()
    return df

# ---------------------------
# MODULE 4: EXPLORATORY DATA ANALYSIS
# ---------------------------
def perform_eda(df, coin_name):
    """Module 3: Comprehensive EDA with insights."""
    st.subheader("📊 Exploratory Data Analysis")
    
    # Basic statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
    price_change_pct = (price_change / df['Close'].iloc[0]) * 100
    avg_volume = df['Volume'].mean()
    volatility = df['Return'].std() * np.sqrt(252) * 100  # Annualized
    
    col1.metric("Current Price", f"${current_price:,.2f}", f"{price_change_pct:.2f}%")
    col2.metric("Price Change", f"${price_change:,.2f}")
    col3.metric("Avg Daily Volume", f"{avg_volume/1e6:.2f}M")
    col4.metric("Volatility (Annual)", f"{volatility:.2f}%")
    col5.metric("Total Days", len(df))
    
    # Trend Analysis
    with st.expander("📈 Trend Analysis", expanded=True):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Movement with Moving Averages', 'Volume Analysis', 'RSI Indicator'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and MAs
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_7'], name='SMA 7', line=dict(color='orange', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_21'], name='SMA 21', line=dict(color='green', dash='dash')), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Volatility Analysis
    with st.expander("📉 Volatility & Risk Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=df['Date'], y=df['Volatility_7']*100, name='7-Day Volatility', fill='tozeroy'))
            fig_vol.update_layout(title='Rolling Volatility', xaxis_title='Date', yaxis_title='Volatility (%)')
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col2:
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Histogram(x=df['Return']*100, nbinsx=50, name='Return Distribution'))
            fig_ret.update_layout(title='Return Distribution', xaxis_title='Daily Return (%)', yaxis_title='Frequency')
            st.plotly_chart(fig_ret, use_container_width=True)
    
    # Statistical insights
    return {
        'current_price': current_price,
        'price_change_pct': price_change_pct,
        'volatility': volatility,
        'avg_volume': avg_volume,
        'sharpe_ratio': (df['Return'].mean() / df['Return'].std()) * np.sqrt(252) if df['Return'].std() > 0 else 0,
        'max_drawdown': ((df['Close'] - df['Close'].cummax()) / df['Close'].cummax()).min() * 100
    }

# ---------------------------
# MODULE 5: PREDICTIVE MODELING
# ---------------------------
def train_models(X_train, X_test, y_train, y_test):
    """Module 4: Train multiple ML models."""
    models = {}
    predictions = {}
    metrics = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models['Linear Regression'] = lr
    predictions['Linear Regression'] = lr_pred
    metrics['Linear Regression'] = calculate_metrics(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['Random Forest'] = rf
    predictions['Random Forest'] = rf_pred
    metrics['Random Forest'] = calculate_metrics(y_test, rf_pred)
    
    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    models['XGBoost'] = xgb
    predictions['XGBoost'] = xgb_pred
    metrics['XGBoost'] = calculate_metrics(y_test, xgb_pred)
    
    return models, predictions, metrics

def train_arima_model(train_data, test_data, order=(5,1,0)):
    """Train ARIMA model for time series prediction."""
    try:
        model = ARIMA(train_data, order=order)
        fitted_model = model.fit()
        predictions = fitted_model.forecast(steps=len(test_data))
        return predictions, calculate_metrics(test_data, predictions)
    except Exception as e:
        st.warning(f"ARIMA training failed: {e}")
        return None, None

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    # Direction accuracy
    actual_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100 if len(actual_direction) > 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }

# ---------------------------
# MODULE 6: ASSET DISCOVERY & RANKING
# ---------------------------
def rank_cryptocurrencies(data_dict):
    """Module 5: Rank cryptocurrencies based on multiple factors."""
    rankings = []
    
    for coin, df in data_dict.items():
        if len(df) < 50:
            continue
            
        # Calculate metrics
        df_temp = create_technical_indicators(df.copy())
        returns = df_temp['Return'].dropna()
        current_price = df['Close'].iloc[-1]
        price_change_30d = ((df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30] * 100) if len(df) >= 30 else 0
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        volume_trend = df['Volume'].iloc[-7:].mean() / df['Volume'].iloc[-30:-7].mean() if len(df) >= 30 else 1
        
        # Momentum score
        momentum = df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1 if len(df) >= 21 else 0
        
        # Risk-adjusted return
        risk_adj_return = sharpe if volatility > 0 else 0
        
        # Composite score
        score = (price_change_30d * 0.3 + sharpe * 100 * 0.3 + momentum * 100 * 0.2 + 
                 (1/volatility if volatility > 0 else 0) * 10 * 0.2)
        
        rankings.append({
            'Cryptocurrency': coin,
            'Current_Price': current_price,
            '30D_Change_%': price_change_30d,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Momentum': momentum,
            'Volume_Trend': volume_trend,
            'Composite_Score': score
        })
    
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values('Composite_Score', ascending=False)
    return rankings_df

# ---------------------------
# SIDEBAR CONFIGURATION
# ---------------------------
st.sidebar.title("⚙️ System Configuration")

# Module Selection
st.sidebar.header("📋 Active Modules")
module_data_acq = st.sidebar.checkbox("Data Acquisition", value=True, disabled=True)
module_preprocess = st.sidebar.checkbox("Data Pre-processing", value=True)
module_eda = st.sidebar.checkbox("Exploratory Data Analysis", value=True)
module_prediction = st.sidebar.checkbox("Predictive Modeling", value=True)
module_ranking = st.sidebar.checkbox("Asset Discovery & Ranking", value=True)

st.sidebar.divider()

# Data Selection
st.sidebar.header("📊 Data Configuration")
available_coins = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "XRP-USD", "MATIC-USD", "DOT-USD", "AVAX-USD"]
primary_coin = st.sidebar.selectbox("Primary Cryptocurrency", available_coins, index=0)
period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

# Comparative Analysis
compare_mode = st.sidebar.checkbox("Enable Multi-Crypto Comparison")
if compare_mode:
    comparison_coins = st.sidebar.multiselect("Select coins to compare", available_coins, default=[available_coins[0], available_coins[1]])
else:
    comparison_coins = [primary_coin]

st.sidebar.divider()

# Model Configuration
st.sidebar.header("🤖 Model Configuration")
models_to_train = st.sidebar.multiselect(
    "Select Models",
    ["Linear Regression", "Random Forest", "XGBoost", "ARIMA"],
    default=["Linear Regression", "Random Forest", "XGBoost", "ARIMA"]
)

test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
use_cross_validation = st.sidebar.checkbox("Use Time Series Cross-Validation", value=False)

st.sidebar.divider()

# Feature Selection
st.sidebar.header("🔧 Feature Engineering")
use_technical_indicators = st.sidebar.checkbox("Include Technical Indicators", value=True)
feature_selection_method = st.sidebar.selectbox("Feature Selection", ["All Features", "Top 10 Features", "Custom"])

# Run Analysis Button
run_analysis = st.sidebar.button("🚀 Run Complete Analysis", type="primary", use_container_width=True)

# ---------------------------
# MAIN APPLICATION LOGIC
# ---------------------------
if run_analysis:
    with st.spinner("🔄 Loading and processing data..."):
        # Load data
        data_dict = load_multiple_cryptos(comparison_coins, period)
        
        if not data_dict or primary_coin not in data_dict:
            st.error("Failed to load data. Please check your internet connection.")
            st.stop()
        
        df_primary = data_dict[primary_coin]
        
        # Module 2: Pre-processing
        if module_preprocess:
            df_primary, outliers_detected = preprocess_data(df_primary)
            st.success(f"✅ Data pre-processing complete. Outliers detected: {outliers_detected}")
        
        # Feature Engineering
        if use_technical_indicators:
            df_primary = create_technical_indicators(df_primary)
            st.success(f"✅ Technical indicators created. Total features: {len(df_primary.columns)}")
        
        # Module 3: EDA
        if module_eda:
            insights = perform_eda(df_primary, primary_coin)
            
            # Display key insights
            st.markdown("### 🎯 Key Insights")
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.metric("Sharpe Ratio", f"{insights['sharpe_ratio']:.2f}")
                st.caption("Risk-adjusted return measure")
                if insights['sharpe_ratio'] > 1:
                    st.success("✅ Good risk-adjusted returns")
                elif insights['sharpe_ratio'] > 0:
                    st.warning("⚠️ Moderate risk-adjusted returns")
                else:
                    st.error("❌ Poor risk-adjusted returns")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.metric("Max Drawdown", f"{insights['max_drawdown']:.2f}%")
                st.caption("Maximum peak-to-trough decline")
                if insights['max_drawdown'] > -20:
                    st.success("✅ Low drawdown risk")
                elif insights['max_drawdown'] > -40:
                    st.warning("⚠️ Moderate drawdown risk")
                else:
                    st.error("❌ High drawdown risk")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with insight_col3:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.metric("Volatility", f"{insights['volatility']:.2f}%")
                st.caption("Annualized volatility")
                if insights['volatility'] < 50:
                    st.success("✅ Relatively stable")
                elif insights['volatility'] < 100:
                    st.warning("⚠️ Moderate volatility")
                else:
                    st.error("❌ High volatility")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Module 4: Predictive Modeling
        if module_prediction and len(df_primary) >= 100:
            st.subheader("🤖 Predictive Modeling & Forecasting")
            
            # Prepare features
            feature_columns = ['Open', 'High', 'Low', 'Volume', 'Return', 'Volatility_7', 
                             'SMA_7', 'SMA_21', 'RSI', 'MACD', 'BB_Width', 'Momentum_7',
                             'Volume_Ratio', 'Daily_Range', 'ATR']
            
            feature_columns = [col for col in feature_columns if col in df_primary.columns]
            
            X = df_primary[feature_columns].values
            y = df_primary['Close'].values
            dates = df_primary['Date'].values
            
            # Train-test split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            dates_train, dates_test = dates[:split_idx], dates[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            with st.spinner("Training machine learning models..."):
                models, predictions, metrics = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
                
                # Train ARIMA if selected
                if "ARIMA" in models_to_train:
                    arima_pred, arima_metrics = train_arima_model(y_train, y_test)
                    if arima_pred is not None:
                        predictions['ARIMA'] = arima_pred
                        metrics['ARIMA'] = arima_metrics
            
            # Display model performance
            st.markdown("### 📊 Model Performance Comparison")
            
            metrics_df = pd.DataFrame(metrics).T
            metrics_df = metrics_df.round(4)
            
            # Display metrics in columns
            metric_cols = st.columns(len(metrics))
            for idx, (model_name, model_metrics) in enumerate(metrics.items()):
                with metric_cols[idx]:
                    st.markdown(f"**{model_name}**")
                    st.metric("RMSE", f"${model_metrics['RMSE']:,.2f}")
                    st.metric("MAE", f"${model_metrics['MAE']:,.2f}")
                    st.metric("R² Score", f"{model_metrics['R2']:.4f}")
                    st.metric("MAPE", f"{model_metrics['MAPE']:.2f}%")
                    st.metric("Direction Accuracy", f"{model_metrics['Direction_Accuracy']:.1f}%")
            
            # Detailed metrics table
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model identification
            best_model = min(metrics.items(), key=lambda x: x[1]['RMSE'])[0]
            st.success(f"🏆 Best Performing Model: **{best_model}** (Lowest RMSE)")
            
            # Visualization
            st.markdown("### 📈 Prediction Visualization")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_test, y=y_test, name='Actual Price', 
                                    line=dict(color='blue', width=2)))
            
            colors = ['red', 'green', 'orange', 'purple']
            for idx, (model_name, pred) in enumerate(predictions.items()):
                fig.add_trace(go.Scatter(x=dates_test, y=pred, name=f'{model_name} Prediction',
                                        line=dict(color=colors[idx % len(colors)], width=2, dash='dash')))
            
            fig.update_layout(
                title=f'{primary_coin} Price Prediction Comparison',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                height=600,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (for tree-based models)
            if 'Random Forest' in models:
                st.markdown("### 🔍 Feature Importance Analysis")
                
                rf_model = models['Random Forest']
                feature_importance = pd.DataFrame({
                    'Feature': feature_columns,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_imp = px.bar(feature_importance.head(10), x='Importance', y='Feature', 
                                orientation='h', title='Top 10 Most Important Features (Random Forest)')
                st.plotly_chart(fig_imp, use_container_width=True)
            
            # Prediction insights
            st.markdown("### 💡 Prediction Insights & Recommendations")
            
            last_actual = y_test[-1]
            predictions_last = {name: pred[-1] for name, pred in predictions.items()}
            avg_prediction = np.mean(list(predictions_last.values()))
            prediction_variance = np.std(list(predictions_last.values()))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${last_actual:,.2f}")
            col2.metric("Avg Model Prediction", f"${avg_prediction:,.2f}", 
                       f"{((avg_prediction - last_actual) / last_actual * 100):+.2f}%")
            col3.metric("Prediction Variance", f"${prediction_variance:,.2f}")
            
            if avg_prediction > last_actual * 1.02:
                st.success("📈 **Bullish Signal**: Models predict price increase. Consider buying opportunities.")
            elif avg_prediction < last_actual * 0.98:
                st.error("📉 **Bearish Signal**: Models predict price decrease. Exercise caution.")
            else:
                st.info("➡️ **Neutral Signal**: Models predict sideways movement. Monitor closely.")
            
            if prediction_variance / last_actual > 0.05:
                st.warning("⚠️ **High Model Disagreement**: Large variance in predictions suggests uncertainty.")
        
        # Module 5: Asset Discovery & Ranking
        if module_ranking and len(data_dict) > 1:
            st.subheader("🏆 Cryptocurrency Ranking & Discovery")
            
            with st.spinner("Analyzing and ranking cryptocurrencies..."):
                rankings_df = rank_cryptocurrencies(data_dict)
            
            st.markdown("### 📊 Comprehensive Ranking Table")
            st.dataframe(rankings_df.style.background_gradient(subset=['Composite_Score'], cmap='RdYlGn'), use_container_width=True)
            
            # Top performers
            st.markdown("### 🌟 Top Performers")
            top_3 = rankings_df.head(3)
            
            top_cols = st.columns(3)
            for idx, (_, row) in enumerate(top_3.iterrows()):
                with top_cols[idx]:
                    st.markdown(f"**#{idx+1} {row['Cryptocurrency']}**")
                    st.metric("Price", f"${row['Current_Price']:,.2f}")
                    st.metric("30D Change", f"{row['30D_Change_%']:.2f}%")
                    st.metric("Sharpe Ratio", f"{row['Sharpe_Ratio']:.3f}")
                    st.metric("Score", f"{row['Composite_Score']:.2f}")
            
            # Ranking visualization
            fig_rank = px.bar(rankings_df, x='Cryptocurrency', y='Composite_Score',
                            title='Cryptocurrency Composite Score Comparison',
                            color='Composite_Score', color_continuous_scale='Viridis')
            st.plotly_chart(fig_rank, use_container_width=True)
            
            # Correlation heatmap
            if len(data_dict) >= 3:
                st.markdown("### 🔗 Cross-Cryptocurrency Correlation Analysis")
                
                # Create correlation matrix
                price_data = pd.DataFrame()
                for coin, df in data_dict.items():
                    price_data[coin] = df.set_index('Date')['Close']
                
                corr_matrix = price_data.corr()
                
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto='.2f',
                                    aspect='auto',
                                    color_continuous_scale='RdBu_r',
                                    title='Price Correlation Matrix')
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Insights
                st.markdown("#### 💡 Correlation Insights")
                high_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > 0.8:
                            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                
                if high_corr:
                    st.info(f"🔗 Found {len(high_corr)} highly correlated pairs (>0.8)")
                    for coin1, coin2, corr_val in high_corr[:5]:
                        st.write(f"- **{coin1}** ↔ **{coin2}**: {corr_val:.3f}")
        
        # Additional Analysis Modules
        st.markdown("---")
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment & Portfolio Insights")
        
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            st.markdown("#### 📊 Value at Risk (VaR) Analysis")
            returns = df_primary['Return'].dropna()
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            st.metric("VaR (95%)", f"{var_95:.2f}%", help="Maximum expected loss at 95% confidence")
            st.metric("VaR (99%)", f"{var_99:.2f}%", help="Maximum expected loss at 99% confidence")
            
            if var_95 < -5:
                st.error("❌ High risk: Significant potential daily losses")
            elif var_95 < -3:
                st.warning("⚠️ Moderate risk: Notable volatility")
            else:
                st.success("✅ Lower risk: Relatively stable")
        
        with risk_col2:
            st.markdown("#### 📈 Distribution Analysis")
            
            # Normality test
            statistic, p_value = stats.normaltest(returns)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            st.write(f"**Skewness**: {skewness:.3f}")
            st.write(f"**Kurtosis**: {kurtosis:.3f}")
            st.write(f"**Normality p-value**: {p_value:.4f}")
            
            if p_value < 0.05:
                st.warning("⚠️ Returns are not normally distributed")
            else:
                st.info("✅ Returns follow normal distribution")
            
            if abs(skewness) > 1:
                st.write(f"{'📉 Negative skew (left tail)' if skewness < 0 else '📈 Positive skew (right tail)'}")
            
            if kurtosis > 3:
                st.write("📊 High kurtosis: Fat tails (extreme events more likely)")
        
        # Market Regime Detection
        st.markdown("---")
        st.subheader("🔄 Market Regime Detection")
        
        # Simple regime detection using moving averages
        df_primary['Regime'] = 'Neutral'
        df_primary.loc[df_primary['Close'] > df_primary['SMA_50'], 'Regime'] = 'Bullish'
        df_primary.loc[df_primary['Close'] < df_primary['SMA_50'], 'Regime'] = 'Bearish'
        
        current_regime = df_primary['Regime'].iloc[-1]
        regime_counts = df_primary['Regime'].value_counts()
        
        regime_col1, regime_col2, regime_col3 = st.columns(3)
        
        with regime_col1:
            st.metric("Current Regime", current_regime)
            if current_regime == 'Bullish':
                st.success("🐂 Bull Market Active")
            elif current_regime == 'Bearish':
                st.error("🐻 Bear Market Active")
            else:
                st.info("➡️ Sideways Market")
        
        with regime_col2:
            bullish_pct = (regime_counts.get('Bullish', 0) / len(df_primary)) * 100
            st.metric("Bullish Days", f"{bullish_pct:.1f}%")
        
        with regime_col3:
            bearish_pct = (regime_counts.get('Bearish', 0) / len(df_primary)) * 100
            st.metric("Bearish Days", f"{bearish_pct:.1f}%")
        
        # Regime visualization
        fig_regime = go.Figure()
        
        for regime, color in [('Bullish', 'green'), ('Bearish', 'red'), ('Neutral', 'gray')]:
            regime_data = df_primary[df_primary['Regime'] == regime]
            if len(regime_data) > 0:
                fig_regime.add_trace(go.Scatter(
                    x=regime_data['Date'],
                    y=regime_data['Close'],
                    mode='markers',
                    name=regime,
                    marker=dict(color=color, size=4)
                ))
        
        fig_regime.update_layout(
            title='Market Regime Classification Over Time',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400
        )
        st.plotly_chart(fig_regime, use_container_width=True)
        
        # Trading Signals
        st.markdown("---")
        st.subheader("📡 Trading Signals & Recommendations")
        
        # Generate signals based on technical indicators
        latest = df_primary.iloc[-1]
        signals = []
        
        # RSI Signal
        if latest['RSI'] < 30:
            signals.append(("🟢 BUY", "RSI Oversold", f"RSI at {latest['RSI']:.1f} (below 30)"))
        elif latest['RSI'] > 70:
            signals.append(("🔴 SELL", "RSI Overbought", f"RSI at {latest['RSI']:.1f} (above 70)"))
        else:
            signals.append(("🟡 HOLD", "RSI Neutral", f"RSI at {latest['RSI']:.1f}"))
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append(("🟢 BUY", "MACD Bullish", "MACD above signal line"))
        else:
            signals.append(("🔴 SELL", "MACD Bearish", "MACD below signal line"))
        
        # Moving Average Signal
        if latest['Close'] > latest['SMA_50']:
            signals.append(("🟢 BUY", "Price Above MA", f"Price ${latest['Close']:.2f} > SMA50 ${latest['SMA_50']:.2f}"))
        else:
            signals.append(("🔴 SELL", "Price Below MA", f"Price ${latest['Close']:.2f} < SMA50 ${latest['SMA_50']:.2f}"))
        
        # Bollinger Bands Signal
        if latest['Close'] < latest['BB_Lower']:
            signals.append(("🟢 BUY", "Below Lower Band", "Potential bounce opportunity"))
        elif latest['Close'] > latest['BB_Upper']:
            signals.append(("🔴 SELL", "Above Upper Band", "Potential reversal"))
        
        # Display signals
        signal_cols = st.columns(len(signals))
        for idx, (signal, indicator, description) in enumerate(signals):
            with signal_cols[idx]:
                st.markdown(f"**{indicator}**")
                st.markdown(f"### {signal}")
                st.caption(description)
        
        # Consolidated recommendation
        buy_signals = sum(1 for s in signals if "BUY" in s[0])
        sell_signals = sum(1 for s in signals if "SELL" in s[0])
        
        st.markdown("---")
        st.markdown("### 🎯 Consolidated Recommendation")
        
        if buy_signals > sell_signals:
            st.success(f"✅ **OVERALL: BULLISH** ({buy_signals} buy signals vs {sell_signals} sell signals)")
            st.write("- Multiple indicators suggest upward momentum")
            st.write("- Consider accumulation or holding positions")
            st.write("- Set stop-loss at recent support levels")
        elif sell_signals > buy_signals:
            st.error(f"❌ **OVERALL: BEARISH** ({sell_signals} sell signals vs {buy_signals} buy signals)")
            st.write("- Multiple indicators suggest downward pressure")
            st.write("- Consider reducing exposure or taking profits")
            st.write("- Wait for better entry points")
        else:
            st.info(f"➡️ **OVERALL: NEUTRAL** ({buy_signals} buy signals, {sell_signals} sell signals)")
            st.write("- Mixed signals indicate consolidation")
            st.write("- Wait for clearer directional bias")
            st.write("- Monitor key support/resistance levels")
        
        # Export Options
        st.markdown("---")
        st.subheader("📥 Export & Reports")
        
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Export raw data
            csv_data = df_primary.to_csv(index=False)
            st.download_button(
                label="📊 Download Raw Data (CSV)",
                data=csv_data,
                file_name=f"{primary_coin}_data_{period}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # Export predictions
            if module_prediction and 'predictions' in locals():
                pred_df = pd.DataFrame({
                    'Date': dates_test,
                    'Actual': y_test,
                    **{f'{name}_Prediction': pred for name, pred in predictions.items()}
                })
                pred_csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="🤖 Download Predictions (CSV)",
                    data=pred_csv,
                    file_name=f"{primary_coin}_predictions_{period}.csv",
                    mime="text/csv"
                )
        
        with export_col3:
            # Export rankings
            if module_ranking and 'rankings_df' in locals():
                rank_csv = rankings_df.to_csv(index=False)
                st.download_button(
                    label="🏆 Download Rankings (CSV)",
                    data=rank_csv,
                    file_name=f"crypto_rankings_{period}.csv",
                    mime="text/csv"
                )
        
        # System Summary
        st.markdown("---")
        st.subheader("📋 Analysis Summary Report")
        
        summary_text = f"""
        ## Cryptocurrency Analysis Report
        **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Primary Asset**: {primary_coin}
        **Period**: {period}
        **Data Points**: {len(df_primary)}
        
        ### Key Metrics
        - Current Price: ${insights['current_price']:,.2f}
        - Period Change: {insights['price_change_pct']:,.2f}%
        - Annualized Volatility: {insights['volatility']:.2f}%
        - Sharpe Ratio: {insights['sharpe_ratio']:.2f}
        - Max Drawdown: {insights['max_drawdown']:.2f}%
        
        ### Market Assessment
        - Current Regime: {current_regime}
        - RSI Level: {latest['RSI']:.1f}
        - Buy Signals: {buy_signals}
        - Sell Signals: {sell_signals}
        
        ### Model Performance (Best: {best_model if module_prediction else 'N/A'})
        """
        
        if module_prediction:
            for model_name, model_metrics in metrics.items():
                summary_text += f"\n- {model_name}: RMSE=${model_metrics['RMSE']:,.2f}, R²={model_metrics['R2']:.4f}"
        
        summary_text += f"""
        
        ### Risk Assessment
        - VaR (95%): {var_95:.2f}%
        - VaR (99%): {var_99:.2f}%
        - Distribution Skewness: {skewness:.3f}
        - Distribution Kurtosis: {kurtosis:.3f}
        
        ### Disclaimer
        This analysis is for informational purposes only and does not constitute financial advice.
        Cryptocurrency investments carry significant risk. Always conduct your own research and
        consult with financial advisors before making investment decisions.
        """
        
        st.markdown(summary_text)
        
        st.download_button(
            label="📄 Download Full Report (TXT)",
            data=summary_text,
            file_name=f"{primary_coin}_analysis_report_{period}.txt",
            mime="text/plain"
        )

else:
    # Initial state - show welcome message
    st.info("👆 Configure your analysis settings in the sidebar and click **'Run Complete Analysis'** to begin!")
    
    # Show feature overview
    st.markdown("---")
    st.markdown("## 🌟 Platform Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("### 📊 Data Intelligence")
        st.write("- Real-time data acquisition")
        st.write("- Advanced pre-processing")
        st.write("- Missing value handling")
        st.write("- Outlier detection")
        st.write("- Data normalization")
    
    with feature_col2:
        st.markdown("### 🤖 ML & Predictions")
        st.write("- Multiple ML models")
        st.write("- Linear Regression")
        st.write("- Random Forest")
        st.write("- XGBoost")
        st.write("- ARIMA time series")
    
    with feature_col3:
        st.markdown("### 📈 Advanced Analytics")
        st.write("- Technical indicators")
        st.write("- Risk assessment")
        st.write("- Market regime detection")
        st.write("- Trading signals")
        st.write("- Asset ranking")
    
    st.markdown("---")
    st.markdown("## 📚 Technical Indicators Included")
    
    indicators_col1, indicators_col2 = st.columns(2)
    
    with indicators_col1:
        st.markdown("""
        **Trend Indicators:**
        - Simple Moving Averages (SMA 7, 21, 50)
        - Exponential Moving Averages (EMA 12, 26)
        - MACD (Moving Average Convergence Divergence)
        - ADX (Average Directional Index)
        
        **Momentum Indicators:**
        - RSI (Relative Strength Index)
        - Stochastic Oscillator
        - Momentum indicators (7-day, 21-day)
        """)
    
    with indicators_col2:
        st.markdown("""
        **Volatility Indicators:**
        - Bollinger Bands
        - ATR (Average True Range)
        - Rolling volatility (multiple windows)
        
        **Volume Indicators:**
        - Volume moving averages
        - Volume ratios
        - On-Balance Volume trends
        """)
    
    st.markdown("---")
    st.markdown("""
    ## ⚠️ Disclaimer
    This platform is designed for **educational and research purposes only**. 
    
    - Past performance does not guarantee future results
    - Cryptocurrency markets are highly volatile and risky
    - Always conduct thorough research before investing
    - Consult with qualified financial advisors
    - Never invest more than you can afford to lose
    
    **The creators of this platform are not liable for any financial losses incurred through its use.**
    """)