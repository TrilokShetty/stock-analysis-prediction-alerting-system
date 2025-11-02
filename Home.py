# main_app.py
import streamlit as st

st.set_page_config(
    page_title="Stock Analysis Dashboard",
    layout="wide"
)

st.title("Stock Analysis, Forecasting, and Price Alerts")

st.markdown("""
Welcome to your financial dashboard. This tool is designed for investors and traders to analyze National Stock Exchange (NSE) listed stocks.

### Features:
- **Fundamental Information**: Get detailed company info and key financial metrics.
- **Next-Day Forecasting**: A machine learning model (LSTM) to forecast the next trading day's closing price.
- **Price Alerts**: Set real-time price alerts for your favorite stocks and get email notifications.

### How to Use:
1.  **Select a Page**: Use the navigation menu in the sidebar to choose a feature.
2.  **Enter a Ticker**: On any page, type an NSE stock ticker symbol (e.g., `RELIANCE.NS`, `TCS.NS`) into the text box and press Enter.
3.  **Analyze**: The page will load all the relevant data, predictions, or alert settings.
""")

st.info("Please select a page from the sidebar to begin.")
st.info("To know more about how to use this app [Click Here](https://github.com/TrilokShetty/stock-analysis-prediction-alerting-system/tree/main)")