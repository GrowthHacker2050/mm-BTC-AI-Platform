import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import datetime

st.set_page_config(layout="wide", page_title="Matt's AI BTC/USD Dashboard", initial_sidebar_state="expanded")
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to", ["Dashboard", "Market Intel", "Trade Notes", "Trade Planner", "Settings"])

exchange = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1h'
limit = 100
import time
for attempt in range(3):
   try:
       ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
       break
   except Exception as e:
       st.warning(f"Attempt {attempt+1}: Failed to fetch data, retrying...")
       time.sleep(2)
else:
   st.error("Failed to load data from Binance after 3 attempts. Please refresh.")
   st.stop()
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
confidence = round(model.predict_proba(latest).max() * 100, 2)

if section == "Dashboard":
    st.title("📊 BTC/USD AI Dashboard")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA20'], line=dict(color='blue'), name='EMA20'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['EMA50'], line=dict(color='orange'), name='EMA50'))
        fig.update_layout(title="BTC/USD K-line Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.header("🔮 AI Trade Signal")
        st.markdown(f"### {signal}")
        st.metric("Confidence", f"{confidence}%")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.metric("Model Accuracy", f"{report['accuracy'] * 100:.2f}%")

elif section == "Market Intel":
    st.title("🧠 Market Intelligence")
    st.write("This page will show crypto market sentiment, news, and generative AI summaries (to be added).")

elif section == "Trade Notes":
    st.title("📝 Trade Notes")
    note = st.text_area("Write your notes:")
    if st.button("Save Note"):
        with open("notes.txt", "a") as f:
            f.write(f"{datetime.datetime.now()}: {note}\n")
        st.success("Note saved.")

elif section == "Trade Planner":
    st.title("📌 Trade Planner")
    balance = st.slider("Your trading capital ($)", 4000, 7000, 5000)
    win_rate = st.slider("Estimated win rate (%)", 50, 90, 70)
    rr_ratio = st.slider("Risk/Reward ratio", 1.0, 5.0, 2.0)
    kelly_fraction = ((win_rate / 100) - (1 - (win_rate / 100)) / rr_ratio)
    risk_amount = round(balance * kelly_fraction, 2)
    st.metric("Suggested Risk per Trade", f"${risk_amount}")

elif section == "Settings":
    st.title("⚙️ Settings")
    st.write("Configure alerts and integration settings here (e.g., Discord, Pushover). [Coming soon]")
