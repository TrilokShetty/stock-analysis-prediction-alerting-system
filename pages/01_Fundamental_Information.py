# pages/01_Fundamental_Information.py
import streamlit as st
import yfinance as yf
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title('Fundamental Information')

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

# Stock Selection
default_ticker = 'RELIANCE.NS'
default_index = symbol_list.index(default_ticker) if default_ticker in symbol_list else 0
ticker = st.selectbox(
    'Enter or Choose NSE listed Stock Symbol', 
    symbol_list, 
    index=default_index
)

if ticker:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        st.subheader(info['longName'])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('**Sector**: ' + info.get('sector', 'N/A'))
            st.markdown('**Industry**: ' + info.get('industry', 'N/A'))
            st.markdown('**Phone**: ' + info.get('phone', 'N/A'))
        with col2:
            st.markdown('**Address**: ' + info.get('address1', 'N/A') + ', ' + info.get('city', 'N/A') + ', ' + info.get('zip', 'N/A') + ', ' + info.get('country', 'N/A'))
            st.markdown('**Website**: ' + info.get('website', 'N/A'))
        
        with st.expander('See detailed business summary'):
            st.write(info.get('longBusinessSummary', 'No summary available.'))

        # --- Stock Price Chart ---
        st.subheader('Stock Price Chart')
        
        min_value = dt.datetime.today() - dt.timedelta(10 * 365)
        max_value = dt.datetime.today()

        col1, col2 = st.columns(2)
        with col1:
            start_input = st.date_input(
                'Enter starting date',
                value=dt.datetime.today() - dt.timedelta(365),
                min_value=min_value, max_value=max_value,
                help='Enter the starting date for the price chart'
            )
        with col2:
            end_input = st.date_input(
                'Enter last date',
                value=dt.datetime.today(),
                min_value=min_value, max_value=max_value,
                help='Enter the last date for the price chart'
            )

        # Download data using yfinance
        hist_price = yf.download(ticker, start_input, end_input)
        
        
        if isinstance(hist_price.columns, pd.MultiIndex):
            hist_price = hist_price.xs(ticker, axis=1, level=1)


        if hist_price.empty:
            st.error("Could not download price data. Check the ticker or date range.")
        else:
            hist_price = hist_price.reset_index()
            hist_price['Date'] = pd.to_datetime(hist_price['Date'])

            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')

            historical_csv = convert_df(hist_price)
            st.download_button(
                label="Download historical data as CSV",
                data=historical_csv,
                file_name=f'{ticker}_historical_data.csv',
                mime='text/csv',
            )

            chart_type = st.radio("Choose Chart Style", ('Candlestick', 'Line Chart'), horizontal=True)
            
            fig = go.Figure()
            
            if chart_type == 'Candlestick':
                fig.add_trace(
                    go.Candlestick(
                        x=hist_price['Date'],
                        open=hist_price['Open'],
                        high=hist_price['High'],
                        low=hist_price['Low'],
                        close=hist_price['Close'],
                        name='OHLC'
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=hist_price['Date'],
                        y=hist_price['Close'],
                        name='Adjusted Close',
                        line=dict(color='blue')
                    )
                )

            fig.update_layout(
                title={'text': f'Stock Prices of {ticker}', 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
                height=600,
                template='plotly_white',
                xaxis_rangeslider_visible=False
            )
            fig.update_yaxes(tickprefix='₹') # Assuming NSE, so Rupees
            st.plotly_chart(fig, use_container_width=True)

        #Post Chart section

        #Financials
        st.subheader('Quarterly Results')
        quarterly_results = stock.quarterly_financials
        if not quarterly_results.empty:
            quarterly_results.columns = quarterly_results.columns.date
            st.dataframe(quarterly_results)
        else:
            st.info("No quarterly results data available.")

        st.subheader('Annual Profit & Loss')
        financials = stock.financials
        if not financials.empty:
            financials.columns = financials.columns.date
            st.dataframe(financials)
        else:
            st.info("No annual P&L data available.")

        # Balance Sheet
        st.subheader('Balance Sheet')
        balance = stock.balance_sheet
        if not balance.empty:
            balance.columns = balance.columns.date
            st.dataframe(balance)
        else:
            st.info("No balance sheet data available.")

        # CashFlows
        st.subheader('Cash Flows')
        cf = stock.cashflow
        if not cf.empty:
            cf.columns = cf.columns.date
            st.dataframe(cf)
        else:
            st.info("No cash flow data available.")
            
    except Exception as e:
        st.error(f"Could not retrieve data for {ticker}. Please check the symbol or try again.")
        st.error(f"Error: {e}")