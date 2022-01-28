import time

import cufflinks as cf
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from gwpy.timeseries import TimeSeries
from matplotlib.backends.backend_agg import RendererAgg
# Make Plotly work in your Jupyter Notebook
# from plotly.offline import init_notebook_mode
from datetime import datetime

import os
from os import listdir
from os.path import isfile, join

import utils as u

# init_notebook_mode(connected=True)
# Use Plotly locally
cf.go_offline()

mpl.use("agg")

_lock = RendererAgg.lock

apptitle = 'Stock market data analysis'

st.set_page_config(page_title=apptitle, page_icon="chart_with_upwards_trend", layout="wide")

# -- Default detector list
detectorlist = ['H1', 'L1', 'V1']

# Title the app
st.title('Stock market data analysis')
st.markdown("""
This Streamlit app is designed to help you analyse stock market data and create the optimal
portfolio for your investment.  \nThe basis for the algorithms come from the repo [Python4Finance](
https://github.com/derekbanas/Python4Finance) by **Derek Banas**.  \n
This app is comprised of three parts which can be selected in the sidebar:  \n
- **sectors** : retrieves stocks from selected sectors ordered by their cumulative return
- **stock** : retrieves information and Ichimoku about a specific Stock
- **portfolio** : computes optimal portfolio using Markowitz optimisation  \n

*This app is also dedicated to Katka Barteková as part of my secret santa gift ;) *""")

st.markdown("---")


@st.cache(ttl=3600, max_entries=10)  # -- Magic command to cache data
def load_gw(t0, detector, fs=4096):
    strain = TimeSeries.fetch_open_data(detector, t0 - 14, t0 + 14, sample_rate=fs, cache=False)
    return strain


def get_port_val_by_date(date, shares, tickers):
    port_prices = u.merge_df_by_column_name('Close', date,
                                            date, *portfolio_input)
    # Convert from dataframe to Python list
    port_prices = port_prices.values.tolist()
    # Trick that converts a list of lists into a single list
    port_prices = sum(port_prices, [])
    return port_prices


PATH = "all_stocks/"
tickers = u.get_column_from_csv("Wilshire-5000-Stocks.csv", "Ticker")

# Start end date defaults
S_DATE = "2017-02-01"
E_DATE = "2022-12-06"
S_DATE_DT = pd.to_datetime(S_DATE)
E_DATE_DT = pd.to_datetime(E_DATE)

st.sidebar.title("Data from Yahoo Finance")
with open('sync_dates.txt') as f:
    lines = f.readlines()
    st.sidebar.markdown("Last sync : " + str(lines[0]))

st.sidebar.markdown("*Takes up to a few hours!*")
if st.sidebar.button("Get latest data"):

    st.sidebar.markdown("This might take a few hours...")
    my_bar = st.sidebar.progress(0)
    for x in range(0, 3481):
        u.save_to_csv_from_yahoo(PATH, tickers[x])
        my_bar.progress(x / 3481)
    files = [x for x in listdir(PATH) if isfile(join(PATH, x))]
    tickers = [os.path.splitext(x)[0] for x in files]
    my_bar2 = st.sidebar.progress(0)
    i = 0
    for x in tickers:
        try:
            new_df = u.get_stock_df_from_csv(x)
            new_df = u.add_daily_return_to_df(new_df)
            new_df = u.add_cum_return_to_df(new_df)
            new_df = u.add_bollinger_bands(new_df)
            new_df = u.add_Ichimoku(new_df)
            new_df.to_csv(PATH + x + '.csv')
            i += 1
            time.sleep(0.1)
            my_bar2.progress(i / len(tickers))
        except Exception as ex:
            print(ex)
    st.success("Data Updated")

    with open('sync_dates.txt', 'w') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)

st.sidebar.markdown("---")

st.sidebar.title("Pick your analysis")
algorithm = st.sidebar.selectbox("Select your analysis",
                                 ["sectors", "stock", "portfolio"])

st.sidebar.markdown("---")

if algorithm == "sectors":
    st.sidebar.title("Select sectors")

    sectors = ["Industrial", "Healthcare", "Inf. Technology", "Communication", "Staples",
               "Discretionary", "Utilities", "Financials", "Materials", "Real Estate", "Energy"]
    sector1 = st.sidebar.selectbox('Sector 1', sectors, index=0)
    sector2 = st.sidebar.selectbox('Sector 2', sectors, index=1)
    sector3 = st.sidebar.selectbox('Sector 3', sectors, index=2)
    sector4 = st.sidebar.selectbox('Sector 4', sectors, index=3)

    graph_type = st.sidebar.radio("Which type of graph to show?", ('Ichimoku', 'Bollinger Bands'))

    sec_df = pd.read_csv("big_stock_sectors.csv")

    st.markdown("**Loading sector data...**")
    my_bar = st.progress(0)
    sec1 = u.get_cum_ret_for_stocks(sec_df.loc[sec_df['Sector'] == sector1])
    my_bar.progress(25)
    sec2 = u.get_cum_ret_for_stocks(sec_df.loc[sec_df['Sector'] == sector2])
    my_bar.progress(50)
    sec3 = u.get_cum_ret_for_stocks(sec_df.loc[sec_df['Sector'] == sector3])
    my_bar.progress(75)
    sec4 = u.get_cum_ret_for_stocks(sec_df.loc[sec_df['Sector'] == sector4])
    my_bar.progress(100)

    st.markdown(
        '##### Top 10 Stocks with the highest Cumulative Return within the selected sectors')

    col1, col2, col3, col4 = st.columns(4)

    sec1 = sec1.sort_values(by=['CUM_RET'], ascending=False).head(10)
    sec2 = sec2.sort_values(by=['CUM_RET'], ascending=False).head(10)
    sec3 = sec3.sort_values(by=['CUM_RET'], ascending=False).head(10)
    sec4 = sec4.sort_values(by=['CUM_RET'], ascending=False).head(10)

    col1.subheader(sector1)
    col1.dataframe(sec1)
    col2.subheader(sector2)
    col2.dataframe(sec2)
    col3.subheader(sector3)
    col3.dataframe(sec3)
    col4.subheader(sector4)
    col4.dataframe(sec4)

    sec1_ticker = sec1['Ticker'].iloc[0]
    sec2_ticker = sec2['Ticker'].iloc[0]
    sec3_ticker = sec3['Ticker'].iloc[0]
    sec4_ticker = sec4['Ticker'].iloc[0]

    if graph_type == 'Bollinger Bands':
        test_df = u.get_stock_df_from_csv(sec1_ticker)
        fig1 = u.plot_with_boll_bands(test_df, f"{sector1} : {sec1_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec2_ticker)
        fig2 = u.plot_with_boll_bands(test_df, f"{sector2} : {sec2_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec3_ticker)
        fig3 = u.plot_with_boll_bands(test_df, f"{sector3} : {sec3_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec4_ticker)
        fig4 = u.plot_with_boll_bands(test_df, f"{sector4} : {sec4_ticker}", 400)
    else:
        test_df = u.get_stock_df_from_csv(sec1_ticker)
        fig1 = u.get_Ichimoku(test_df, f"{sector1} : {sec1_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec2_ticker)
        fig2 = u.get_Ichimoku(test_df, f"{sector2} : {sec2_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec3_ticker)
        fig3 = u.get_Ichimoku(test_df, f"{sector3} : {sec3_ticker}", 400)

        test_df = u.get_stock_df_from_csv(sec4_ticker)
        fig4 = u.get_Ichimoku(test_df, f"{sector4} : {sec4_ticker}", 400)

    st.markdown("##### Stocks with highest Cumulative Return for each Sector ")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    col3.plotly_chart(fig3, use_container_width=True)
    col4.plotly_chart(fig4, use_container_width=True)
elif algorithm == "stock":
    input_ticker = st.sidebar.multiselect("Tickers to analyse", tickers, default=['TSLA'])
    stock_graph_type = st.sidebar.radio("Which type of graphs to show?", ('Ichimoku', 'Bollinger '
                                                                                      'Bands'))

    for input in input_ticker:

        test_df = u.get_stock_df_from_csv(input)
        # st.markdown(f"### Ichimoku Chart for the {input_ticker} Stock")

        msft = yf.Ticker(input)
        # st.header(msft.info['longName'])

        try:
            st.header(f"[{msft.info['longName']}]({msft.info['website']})")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.markdown(f"{msft.info['country']}, {msft.info['state']}  \n"
                        f"{msft.info['city']}, {msft.info['zip']}  \n")
            c2.image(msft.info["logo_url"], width=40)

            col1, col2, col3 = st.columns(3)
            col1.text_area("Long Business Summary", msft.info['longBusinessSummary'], height=100)

            col2.markdown(" ")
            col2.markdown(" ")

            col2.markdown(f"Industry : *{msft.info['industry']}*  \n"
                          f"Employees : *{msft.info['fullTimeEmployees']}*  \n"
                          f"Gross Profit : *{msft.info['grossProfits']:,}$*  \n"
                          f"Recommendation : *{msft.info['recommendationKey']}*  \n")

            if stock_graph_type == "Ichimoku":
                fig5 = u.get_Ichimoku(test_df, f"{input}", 700)
                st.plotly_chart(fig5, use_container_width=True)
            else:
                fig5 = u.plot_with_boll_bands(test_df, f"{input}", 700)
                st.plotly_chart(fig5, use_container_width=True)

            with st.expander(f"More information about {msft.info['longName']}",
                             expanded=False):
                st.write(msft.info)

            st.markdown("---")

        except KeyError:
            st.error("ERROR : Ticker not found")
else:
    st.header("Markowitz Portfolio Optimization")
    st.markdown("""
    Harry Markowitz proved that you could make what is called an efficient portfolio.
    That is a portfolio that optimizes return while also minimizing risk.
    We don't benefit from analyzing individual securities at the same rate
    as if we instead considered a portfolio of stocks.

    We do this by creating portfolios with stocks that are **not correlated**.
    We want to calculate expected returns by analyzing the returns of each stock multiplied
    by its weight.

    $w_1r_1 + w_2r_2 = r_p$

    The standard deviation of the portfolio is found this way.
    Sum multiple calculations starting by finding the product of the first securities weight
    squared times its standard deviation squared.
    The middle is 2 times the correlation coefficient between the stocks.
    And, finally add those to the weight squared times the standard deviation squared
    for the second security.

    $(w_1\sigma_1 + w_2\sigma_2)^2 = w_1^2\sigma_1^2 + 2w_1\sigma_1w_2\sigma_2\\rho_1 +
    w_2^2\sigma_2^2$""")

    # portfolio_input = st.text_input("Input your portfolio as comma separated tickers",
    #                                 "CALX, NOVT, RGEN, LLY, AMD, NFLX, COST, BJ, WING, MSCI, CBRE")
    portfolio_input = st.multiselect("Input your portfolio as comma separated tickers",
                                     default=['CALX', 'NOVT', 'RGEN', 'LLY', 'AMD', 'NFLX', 'COST',
                                              'BJ', 'WING', 'MSCI', 'CBRE'])
    # portfolio_input = portfolio_input.replace(" ", "")
    # portfolio_input = portfolio_input.split(",")

    col1, col2 = st.columns(2)
    mult_df = u.merge_df_by_column_name('Close', S_DATE, E_DATE, *portfolio_input)

    fig = px.line(mult_df, x=mult_df.index, y=mult_df.columns)
    fig.update_layout(title="Price of Investments over Total Dataset")
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=400, width=1800, showlegend=True)
    col1.plotly_chart(fig, use_container_width=True)

    mult_cum_df = u.merge_df_by_column_name('cum_return', S_DATE, E_DATE, *portfolio_input)

    fig = px.line(mult_cum_df, x=mult_cum_df.index, y=mult_cum_df.columns)
    fig.update_layout(title="Cumulative Return for all Stocks")
    fig.update_xaxes(title="Date", rangeslider_visible=True)
    fig.update_yaxes(title="Price")
    fig.update_layout(height=400, width=1800,
                      showlegend=True)
    col2.plotly_chart(fig, use_container_width=True)

    returns = np.log(mult_df / mult_df.shift(1))
    # TODO: finish somehow mean return
    # mean_ret = returns.mean() * 252  # 252 average trading days per year
    # st.header(f"Mean Return")
    # st.dataframe(mean_ret)

    st.header("Correlation Matrix")
    st.markdown("As we want to avoid correlated Stocks, correlations above 0.5 are highlighted in "
                "red.")
    st.table(
        returns.corr().style.apply(lambda x: ["color : red" if 1.0 > v >= 0.5 else "" for v in x],
                                   axis=1))

    num_stocks = len(portfolio_input)
    risk_free_rate = 0.0125

    p_ret = []  # Returns list
    p_vol = []  # Volatility list
    p_SR = []  # Sharpe Ratio list
    p_wt = []  # Stock weights list

    port_amt = 10000

    st.markdown(f"**Evaluating {port_amt} random portfolios...**")
    my_bar = st.progress(0)

    for x in range(port_amt):
        # Generate random weights
        p_weights = np.random.random(num_stocks)
        p_weights /= np.sum(p_weights)

        # Add return using those weights to list
        ret_1 = np.sum(p_weights * returns.mean()) * 252
        p_ret.append(ret_1)

        # Add volatility or standard deviation to list
        vol_1 = np.sqrt(np.dot(p_weights.T, np.dot(returns.cov() * 252, p_weights)))
        p_vol.append(vol_1)

        # Get Sharpe ratio
        SR_1 = (ret_1 - risk_free_rate) / vol_1
        p_SR.append(SR_1)

        # Store the weights for each portfolio
        p_wt.append(p_weights)

        if x % 100 == 0:
            my_bar.progress(int((x / 100) + 1))

    # Convert to Numpy arrays
    p_ret = np.array(p_ret)
    p_vol = np.array(p_vol)
    p_SR = np.array(p_SR)
    p_wt = np.array(p_wt)

    ports = pd.DataFrame({'Return': p_ret, 'Volatility': p_vol})
    col1, col2, col3 = st.columns(3)
    fig = go.Figure(data=[go.Scatter(x=p_vol, y=p_ret, mode='markers',
                                     marker=dict(size=2, color=p_SR, autocolorscale=True,
                                                 showscale=True))])
    fig.update_layout(title="Efficient Frontier")
    fig.update_xaxes(title="Volatility")
    fig.update_yaxes(title="Expected Return")
    col3.plotly_chart(fig, use_container_width=True)

    SR_idx = np.argmax(p_SR)

    # Find the ideal portfolio weighting at that index
    all_stocks_port = ""
    port_wts = []
    i = 0
    while i < num_stocks:
        port_wts.append(p_wt[SR_idx][i] * 100)
        all_stocks_port += f"**{portfolio_input[i]}** : {round(p_wt[SR_idx][i] * 100, 4)}%  \n"
        i += 1

    col1.header("Optimal Portfolio")
    col1.markdown(all_stocks_port)

    col1.markdown(f"**Volatility : {round(p_vol[SR_idx], 4)}  \nReturn : "
                  f"{round(p_ret[SR_idx], 4)}**  \n"
                  f"*in 1 year (252 trading days)*")

    fig = go.Figure(data=[go.Pie(labels=portfolio_input, values=port_wts)])
    col2.plotly_chart(fig, use_container_width=True)

    # port_wts = [7, 8, 15, 14, 3, 3, 17, 6, 11, 14, 1]

    # # Get all stock prices on the starting date
    # port_df_start = u.merge_df_by_column_name('Close', '2022-01-07', '2022-01-07',
    #                                           *portfolio_input)
    # # Convert from dataframe to Python list
    # port_prices = port_df_start.values.tolist()
    #
    # # Trick that converts a list of lists into a single list
    # port_prices = sum(port_prices, [])
    #
    # tot_shares, share_cost = u.get_port_shares(105.64, True, port_wts, port_prices)
    # st.markdown(f"Shares : {tot_shares}")
    # st.markdown(f"Share Cost : {share_cost}")
    #
    # # Get list of weights for stocks
    # stock_wts = u.get_port_weighting(share_cost)
    # st.markdown(f"Stock Weights : {stock_wts}")
    #
    # # Get value at end of year
    # st.markdown(get_port_val_by_date(E_DATE, tot_shares, portfolio_input))

    # st.markdown("correlation matrix")
    # st.markdown("risk-free rate")
    # st.markdown("expected return")
    # st.markdown("expected variance")
    # st.markdown("expected covariance")
    # st.markdown("expected portfolio variance")
    # st.markdown("expected portfolio covariance")
    # st.markdown("expected portfolio return")
    # st.markdown("expected portfolio risk")
    # st.markdown("expected portfolio sharpe ratio")
    # st.markdown("expected portfolio sortino ratio")
    # st.markdown("expected portfolio information ratio")
    # st.markdown("expected portfolio beta")
    # st.markdown("expected portfolio alpha")
    # st.markdown("expected portfolio max drawdown")
    # st.markdown("expected portfolio max drawdown duration")
    # st.markdown("expected portfolio max drawdown start")
    # st.markdown("expected portfolio max drawdown end")
