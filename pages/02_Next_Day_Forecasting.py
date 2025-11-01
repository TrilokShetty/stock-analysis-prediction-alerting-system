# pages/05_Next_Day_Forecasting.py
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error 
import numpy as np 


np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)

st.title('Next-Day Price Forecasting (LSTM)')
st.write("This page uses a Long Short-Term Memory (LSTM) model, a type of recurrent neural network, to forecast the next trading day's closing price. The model is trained on the fly using the last 5 years of data.")
st.write("LSTM model is chosen as it achieved the lowest RMSE among all compared models, indicating the best prediction accuracy. [Click here](https://github.com/TrilokShetty/stock-analysis-prediction-alerting-system/blob/main/Stock-Prediction.ipynb)")

# Load Symbols
@st.cache_data
def load_symbols():
    try:
        csv = pd.read_csv('symbols.csv')
        symbol = csv['Symbol'].tolist()
        for i in range(0, len(symbol)):
            symbol[i] = symbol[i] + ".NS"
        return symbol
    except FileNotFoundError:
        st.error("`symbols.csv` not found. Please add it to the root directory.")
        return ['RELIANCE.NS'] # Fallback

symbol_list = load_symbols()

# Ticker Selection
default_ticker = 'RELIANCE.NS'
default_index = symbol_list.index(default_ticker) if default_ticker in symbol_list else 0
ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol', 
    symbol_list, 
    index=default_index
)


@st.cache_data
def run_lstm_model(ticker):
    try:
        start = dt.datetime.today() - dt.timedelta(5 * 365)
        end = dt.datetime.today()

        df = yf.download(ticker, start, end)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        if df.empty:
            return None, "Could not download price data."
            
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        data = df.sort_index(ascending=True, axis=0)
        new_data = data[['Date', 'Close']].copy()
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        dataset = new_data.values
        
        # Used 80% of data for training
        train_len = int(len(dataset) * 0.8)
        
        # Handle case where data is too short
        if train_len <= 60: # 60 is the look_back period
            return None, f"Not enough data to train. Need more than {60 / 0.8:.0f} days, but found {len(dataset)}."

        train = dataset[0:train_len, :]
        valid = dataset[train_len:, :]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        look_back = 60 # Look back period
        
        for i in range(look_back, len(train)):
            x_train.append(scaled_data[i - look_back:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # building LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0) # Kept epochs low for speed (can be tweaked)

        
        inputs = new_data[len(new_data) - len(valid) - look_back:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(look_back, inputs.shape[0]):
            X_test.append(inputs[i - look_back:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        closing_price_preds = model.predict(X_test)
        closing_price_preds = scaler.inverse_transform(closing_price_preds)

        
        real_data = [inputs[len(inputs) - look_back:len(inputs), 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)

        
        train_data = new_data[:train_len]
        
        
        valid_data = new_data[train_len:].copy()
        
        
        valid_data['Predictions'] = closing_price_preds

        return (train_data, valid_data, prediction[0][0]), None
        
    except Exception as e:
        return None, f"An error occurred during model training: {e}"


if st.button('Run Forecasting Model', type="primary"):
    if not ticker:
        st.error("Please select a ticker symbol.")
    else:
        with st.spinner('Downloading data, training model, and making predictions... This may take a minute.'):
            result, error = run_lstm_model(ticker)
        
        if error:
            st.error(error)
        else:
            train_data, valid_data, next_day_prediction = result
            
            st.success("Model training and prediction complete!")
            
            st.subheader('Forecasted vs. Actual Price (Test Set)')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], name='Train Data'))
            fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Close'], name='Actual Price'))
            fig.add_trace(go.Scatter(x=valid_data.index, y=valid_data['Predictions'], name='Predicted Price'))
            fig.update_layout(title='Model Performance', xaxis_title='Date', yaxis_title='Close Price',
                                template='plotly_white', height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Metrics - MAE RMSE
            st.subheader('Model Metrics')
            
            actuals = valid_data['Close']
            predictions = valid_data['Predictions']

            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions)) # Calculate RMSE
            
            col1, col2 = st.columns(2)
            col1.metric('Mean Absolute Error (MAE)', f"₹{mae:.2f}")
            col2.metric('Root Mean Squared Error (RMSE)', f"₹{rmse:.2f}")

            #next day forecast
            st.subheader('Next-Day Forecast')
            last_actual_price = valid_data['Close'].iloc[-1]
            delta = next_day_prediction - last_actual_price
            
            st.metric(
                label=f"Forecasted Closing Price for Next Trading Day",
                value=f"{next_day_prediction:.2f}",
                delta=f"{delta:.2f} ({delta / last_actual_price * 100:.2f}%) vs. last close"
            )
            st.caption(f"Last actual close: ₹{last_actual_price:.2f} on {valid_data.index[-1].strftime('%Y-%m-%d')}")