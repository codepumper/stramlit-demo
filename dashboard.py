import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import plotly.express as px

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("Stock Price Prediction with SVM")

# Load data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]  # Keep only the first level

    return data

# Stock selection
stocks = {"Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOG"}
selected_stock = st.selectbox("Select Stock", list(stocks.keys()))
ticker = stocks[selected_stock]

def calculate_indicators(df):
    df["RSI"] = ta.rsi(df.Close, length=14)
    df["SMA_10"] = ta.sma(df.Close, length=10)
    df["SMA_50"] = ta.sma(df.Close, length=50)
    df["EMA_20"] = ta.ema(df.Close, length=20)
    df["CCI"] = ta.cci(df.High, df.Low, df.Close, length=20)
    return df.dropna()

# Persist data across reruns
if "data" not in st.session_state:
    raw_data = load_data(ticker)
    st.session_state["data"] = calculate_indicators(raw_data.copy())

data = st.session_state["data"]  # Use persisted data

print(data)

# Create target variable
data["target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

# Feature selection
features = ["RSI", "SMA_10", "SMA_50", "EMA_20", "CCI"]

st.sidebar.header("Feature Selection")
selected_features = []
for feature in features:
    if st.sidebar.checkbox(feature, value=True, key=feature):
        selected_features.append(feature)

# Train/test split ratio
train_ratio = st.sidebar.slider("Train/Test Ratio", 0.1, 0.9, 0.8, 0.05)

# Model training and evaluation
if st.sidebar.button("▶️ Train New Model"):
    if data.empty:
        st.error("Data is empty. Please reload the page or select another stock.")
        st.stop()

    if len(selected_features) == 0:
        st.error("Please select at least one feature!")
    else:
        # Prepare data
        X = data[selected_features]
        y = data["target"]
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Debugging Output
        st.subheader("Training Features")
        st.write(X_train)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training
        model = SVC()
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate returns
        test_data = data.iloc[split_idx:].copy()
        test_data["Prediction"] = y_pred
        test_data["Strategy_Return"] = test_data["Close"].pct_change().shift(-1) * test_data["Prediction"]
        test_data = test_data.dropna()
        
        # Metrics
        avg_return = test_data["Strategy_Return"].mean()
        cumulative_return = (test_data["Strategy_Return"] + 1).cumprod()

        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Average Daily Return", f"{avg_return:.2%}")

        # Plots
        fig1 = px.line(cumulative_return, title="Cumulative Returns")
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = px.line(test_data["Close"], title="Stock Price During Test Period")
        st.plotly_chart(fig2, use_container_width=True)

# # Display raw data
# if st.checkbox("Show Raw Data"):
#     st.subheader("Raw Data")
#     st.write(data)
