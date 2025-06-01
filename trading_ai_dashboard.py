import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(layout="wide")
st.title("📊 BTC/USDT AI Trading Signal Dashboard")

# Load BTC/USDT data from Binance
exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 100

with st.spinner("Fetching BTC/USDT data..."):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df[:-1]

features = ['Open', 'High', 'Low', 'Close', 'EMA20', 'EMA50']
X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

latest = df[features].iloc[[-1]]
prediction = model.predict(latest)[0]
signal = "📈 BUY (Expected Up)" if prediction == 1 else "📉 SELL (Expected Down)"

col1, col2 = st.columns([2, 1])

with col1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], line=dict(color='blue'), name='EMA20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], line=dict(color='orange'), name='EMA50'))
    fig.update_layout(title="BTC/USDT K-line with EMA", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔮 AI Trade Signal")
    st.markdown(f"### {signal}")
    st.subheader("📈 Model Accuracy")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.metric("Accuracy", f"{report['accuracy'] * 100:.2f}%")
    st.metric("Precision (Down)", f"{report['0']['precision'] * 100:.2f}%")
    st.metric("Precision (Up)", f"{report['1']['precision'] * 100:.2f}%")
