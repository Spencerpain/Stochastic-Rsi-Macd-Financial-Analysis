import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Title and Input from the User
st.title("Stock Analysis Tool")
st.write("This tool analyzes technical indicators for a given stock, using stochastic processes, relative strength index, and moving average convergence/divergence strategies.")
st.markdown("This website is meant as a **suggestion**, so please do your own resarch into your investments")
# Input: Ticker, interval, and start date from the user
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL, TSLA, ETH-USD)", value="ETH-USD")
interval = st.selectbox("Select data interval", ["30m", "1h", "1d"])
start_date = st.date_input("Select the start date (maximum 60 days back for 30m interval)", pd.to_datetime("2024-08-08"))

# Download data from Yahoo Finance and run analysis
if st.button("Run Analysis"):
    st.write(f"Fetching data for {ticker}...")

    # Main code (unaltered from your original script)
    df = yf.download(ticker, interval=interval, start=start_date)

    if df.empty:
        st.write("No data found for the given ticker and date range. Please try again.")
    else:
        # Calculate the technical indicators
        # Manually calculate %K and %D to replace ta.momentum.stoch
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        df['%D'] = df['%K'].rolling(window=3).mean()  # %D is a 3-period moving average of %K

        # Calculate RSI
        delta = df['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))

        # Calculate MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26

        # Drop NaN values resulting from indicator calculations
        df.dropna(inplace=True)

        # Define the function to get trigger signals with counts
        def gettriggers(df, lags, buy=True):
            masks = []
            for i in range(1, lags + 1):
                if buy:
                    mask = (df['%K'].shift(i) < 20) & (df['%D'].shift(i) < 20)
                else:
                    mask = (df['%K'].shift(i) > 80) & (df['%D'].shift(i) > 80)
                masks.append(mask)
            dfx = pd.concat(masks, axis=1)
            return dfx.sum(axis=1)

        # Apply the function to get buy and sell trigger signals
        df['Buytrigger'] = np.where(gettriggers(df, 6), 1, 0)
        df['Selltrigger'] = np.where(gettriggers(df, 6, False), 1, 0)

        # Define buy and sell criteria
        df['Buy'] = np.where((df.Buytrigger) & (df['%K'].between(20, 80)) & df['%D'].between(20, 80) & (df.rsi > 50) & (df.macd > 0), 1, 0)
        df['Sell'] = np.where((df.Selltrigger) & (df['%K'].between(20, 80)) & df['%D'].between(20, 80) & (df.rsi < 50) & (df.macd < 0), 1, 0)

        # Extract buying and selling dates
        Buying_dates, Selling_dates = [], []
        for i in range(len(df) - 1):
            if df.Buy.iloc[i]:
                Buying_dates.append(df.iloc[i + 1].name)
                for num, j in enumerate(df.Sell[i:]):
                    if j:
                        Selling_dates.append(df.iloc[i + num + 1].name)
                        break

        cutit = len(Buying_dates) - len(Selling_dates)
        if cutit:
            Buying_dates = Buying_dates[:-cutit]

        frame = pd.DataFrame({'Buying_dates': Buying_dates, 'Selling_dates': Selling_dates})

        # Filter valid actual transactions
        actuals = frame[frame.Buying_dates > frame.Selling_dates.shift(1)]

        # Calculate profits
        def profitcalc():
            Buyprices = df.loc[actuals.Buying_dates].Open
            Sellprices = df.loc[actuals.Selling_dates].Open
            return (Sellprices.values - Buyprices.values) / Buyprices.values

        profits = profitcalc()
        pp = (profits + 1).prod()

        # Display cumulative profit potential
        st.write("### Summary of Trading Strategy")
        st.write(f"Total Profit Potential (Cumulative Product of Profits): {pp:.2f}")

        # Display Buy and Sell Signals DataFrame
        st.write("### Buy and Sell Signals DataFrame")
        st.write(actuals)

        # Plotting the data
        st.write("### Stock Price with Buy and Sell Signals")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df.index, df['Close'], color='k', alpha=0.7, label='Close Price')
        ax.scatter(actuals.Buying_dates, df.loc[actuals.Buying_dates, 'Open'], marker='^', color='g', s=100, label='Buy Signal')
        ax.scatter(actuals.Selling_dates, df.loc[actuals.Selling_dates, 'Open'], marker='v', color='r', s=100, label='Sell Signal')
        ax.set_title(f"{ticker} Buy and Sell Signals")
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)
