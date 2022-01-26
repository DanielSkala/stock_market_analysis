import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Make Plotly work in your Jupyter Notebook

PATH = "all_stocks/"
# tickers = get_column_from_csv("/Wilshire-5000-Stocks.csv", "Ticker")

# Start end date defaults
S_DATE = "2017-02-01"
E_DATE = "2022-12-06"
S_DATE_DT = pd.to_datetime(S_DATE)
E_DATE_DT = pd.to_datetime(E_DATE)


def get_stock_df_from_csv(ticker):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        df = pd.read_csv(PATH + ticker + '.csv', index_col=0)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df


def get_cum_ret_for_stocks(stock_df):
    tickers = []
    cum_rets = []

    for index, row in stock_df.iterrows():
        df = get_stock_df_from_csv(row['Ticker'])
        if df is None:
            pass
        else:
            tickers.append(row['Ticker'])
            cum = df['cum_return'].iloc[-1]
            cum_rets.append(cum)
    return pd.DataFrame({'Ticker': tickers, 'CUM_RET': cum_rets})


def get_fill_color(label):
    if label >= 1:
        return 'rgba(0,250,0,0.4)'
    else:
        return 'rgba(250,0,0,0.4)'


def get_Ichimoku(df, header, height):
    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df["Low"], close=df['Close'], name="Candlestick")

    df1 = df.copy()
    fig = go.Figure()

    df['label'] = np.where(df['SpanA'] > df['SpanB'], 1, 0)
    df['group'] = df['label'].ne(df['label'].shift()).cumsum()

    df = df.groupby('group')

    dfs = []
    for name, data in df:
        dfs.append(data)

    for df in dfs:
        fig.add_traces(go.Scatter(x=df.index, y=df.SpanA,
                                  line=dict(color='rgba(0,0,0,0)')))

        fig.add_traces(go.Scatter(x=df.index, y=df.SpanB,
                                  line=dict(color='rgba(0,0,0,0)'),
                                  fill='tonexty',
                                  fillcolor=get_fill_color(df['label'].iloc[0])))

    baseline = go.Scatter(x=df1.index, y=df1['Baseline'],
                          line=dict(color='pink', width=2), name="Baseline")

    conversion = go.Scatter(x=df1.index, y=df1['Conversion'],
                            line=dict(color='black', width=1), name="Conversion")

    lagging = go.Scatter(x=df1.index, y=df1['Lagging'],
                         line=dict(color='purple', width=2), name="Lagging")

    span_a = go.Scatter(x=df1.index, y=df1['SpanA'],
                        line=dict(color='green', width=2, dash='dot'), name="Span A")

    span_b = go.Scatter(x=df1.index, y=df1['SpanB'],
                        line=dict(color='red', width=1, dash='dot'), name="Span B")

    fig.add_trace(candle)
    fig.add_trace(baseline)
    fig.add_trace(conversion)
    fig.add_trace(lagging)
    fig.add_trace(span_a)
    fig.add_trace(span_b)

    fig.update_layout(height=height, width=1600, showlegend=True)
    fig.update_layout(title=header)

    # fig.show()
    return fig


def plot_with_boll_bands(df, header, height):
    fig = go.Figure()

    candle = go.Candlestick(x=df.index, open=df['Open'],
                            high=df['High'], low=df['Low'],
                            close=df['Close'], name="Candlestick")

    upper_line = go.Scatter(x=df.index, y=df['upper_band'],
                            line=dict(color='rgba(250, 0, 0, 0.75)',
                                      width=1), name="Upper Band")

    mid_line = go.Scatter(x=df.index, y=df['middle_band'],
                          line=dict(color='rgba(0, 0, 250, 0.75)',
                                    width=0.7), name="Middle Band")

    lower_line = go.Scatter(x=df.index, y=df['lower_band'],
                            line=dict(color='rgba(0, 250, 0, 0.75)',
                                      width=1), name="Lower Band")

    fig.add_trace(candle)
    fig.add_trace(upper_line)
    fig.add_trace(mid_line)
    fig.add_trace(lower_line)

    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")

    # USED FOR NON-DAILY DATA : Get rid of empty dates and market closed
    # fig.update_layout(title=ticker + " Bollinger Bands",
    # height=1200, width=1800,
    #               showlegend=True,
    #               xaxis_rangebreaks=[
    #         dict(bounds=["sat", "mon"]),
    #         dict(bounds=[16, 9.5], pattern="hour"),
    #         dict(values=["2021-12-25", "2022-01-01"])
    #     ])

    fig.update_layout(height=height, width=1600, showlegend=True)
    fig.update_layout(title=header)

    # fig.show()
    return fig


def merge_df_by_column_name(col_name, sdate, edate, *tickers):
    # Will hold data for all dataframes with the same column name
    mult_df = pd.DataFrame()

    for x in tickers:
        df = get_stock_df_from_csv(x)

        # NEW Check if your dataframe has duplicate indexes
        if not df.index.is_unique:
            # Delete duplicates
            df = df.loc[~df.index.duplicated(), :]

        mask = (df.index >= sdate) & (df.index <= edate)
        mult_df[x] = df.loc[mask][col_name]

    return mult_df


def get_port_shares(one_price, force_one, wts, prices):
    # Gets number of stocks to analyze
    num_stocks = len(wts)

    # Holds the number of shares for each
    shares = []

    # Holds Cost of shares for each
    cost_shares = []

    i = 0
    while i < num_stocks:
        # Get max amount to spend on stock
        max_price = one_price * wts[i]

        # Gets number of shares to buy and adds them to list
        num_shares = int(max_price / prices[i])

        # If the user wants to force buying one share do it
        if (force_one & (num_shares == 0)):
            num_shares = 1

        shares.append(num_shares)

        # Gets cost of those shares and appends to list
        cost = num_shares * prices[i]
        cost_shares.append(cost)
        i += 1

    return shares, cost_shares


def get_port_weighting(share_cost):
    # Holds weights for stocks
    stock_wts = []
    # All values summed
    tot_val = sum(share_cost)
    print("Total Investment :", tot_val)

    for x in share_cost:
        stock_wts.append(x / tot_val)
    return stock_wts


def get_column_from_csv(file, col_name):
    # Try to get the file and if it doesn't exist issue a warning
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File Doesn't Exist")
    else:
        return df[col_name]


def add_daily_return_to_df(df):
    df['daily_return'] = (df['Close'] / df['Close'].shift(1)) - 1
    # Save data to a CSV file
    # df.to_csv(PATH + ticker + '.csv')
    return df


def add_cum_return_to_df(df):
    df['cum_return'] = (1 + df['daily_return']).cumprod()
    # df.to_csv(PATH + ticker + '.csv')
    return df


def add_bollinger_bands(df):
    df['middle_band'] = df['Close'].rolling(window=20).mean()
    df['upper_band'] = df['middle_band'] + 1.96 * df['Close'].rolling(window=20).std()
    df['lower_band'] = df['middle_band'] - 1.96 * df['Close'].rolling(window=20).std()
    # df.to_csv(PATH + ticker + '.csv')
    return df


def add_Ichimoku(df):
    # Conversion
    hi_val = df['High'].rolling(window=9).max()
    low_val = df['Low'].rolling(window=9).min()
    df['Conversion'] = (hi_val + low_val) / 2

    # Baseline
    hi_val2 = df['High'].rolling(window=26).max()
    low_val2 = df['Low'].rolling(window=26).min()
    df['Baseline'] = (hi_val2 + low_val2) / 2

    # Spans
    df['SpanA'] = ((df['Conversion'] + df['Baseline']) / 2).shift(26)
    hi_val3 = df['High'].rolling(window=52).max()
    low_val3 = df['Low'].rolling(window=52).min()
    df['SpanB'] = ((hi_val3 + low_val3) / 2).shift(26)
    df['Lagging'] = df['Close'].shift(-26)

    return df


def save_to_csv_from_yahoo(folder, ticker):
    stock = yf.Ticker(ticker)
    try:
        df = stock.history(period="5y")
        the_file = folder + ticker.replace(".", "_") + '.csv'
        df.to_csv(the_file)
    except Exception:
        st.error(f"Couldn't Get Data for : {ticker}")
