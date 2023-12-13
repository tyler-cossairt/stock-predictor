import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


st.title('Stock Predictions')

startDate = '2013-01-01'
endDate = date.today().strftime('%Y-%m-%d')

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        stock = st.text_input('', 'AAPL')
        tickerData = yf.Ticker(stock)
        tickerDf = tickerData.history(period='1d', start=startDate, end=endDate)
    with col2:
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365

st.write('Stock: ','(', tickerData.info['longName'], ')')

def loadData(ticker):
    data = yf.download(ticker, startDate, endDate)
    data.reset_index(inplace=True)
    return data

with st.container(border=True):
    col1, col2 = st.columns([1,2])
    with col1:
        data_load_state = st.text('Loading data...')
        data = loadData(stock)
        st.write(data.tail(10))
        data_load_state.text('Loading data... done!')

    with col2:
        def plotRawData():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
            fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        plotRawData()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

with st.container(border=True):
    st.subheader('Forecast data')
    st.write(forecast.tail(10))

with st.container(border=True):
    st.subheader('Forecast time series')
    fig1 = plot_plotly(m, forecast, xlabel='Date', ylabel='Price')
    st.plotly_chart(fig1, use_container_width=True, theme=None)

with st.container(border=True):
    st.subheader('Forecast components')
    fig3 = plot_components_plotly(m, forecast, figsize=(900, 450))
    st.plotly_chart(fig3, theme=None, use_container_width=True)

