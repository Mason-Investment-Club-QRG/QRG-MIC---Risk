# Sharpe Ratio File - Quantitative Risk Group (Risk Analyst: Abishek Samuel)
# Calculate the Portfolio Sharpe Ratio
# Using Yfinance for returns and then traditional Sharpe Ratio Formula
# SR Displayed

import pandas as pd             # for large datasets and intrepret data
import numpy as np              # for numerical analysis and computations
import matplotlib.pyplot as plt # for graphs
import yfinance as yf           # used for financial data (pulled from Yahoo Finance)
import math                     # for mathmematical calculations 

# Standard Variables
trading_days = 252
holding_weights = [0.0958, 0.1091, 0.1111, 0.0386, 0.0447, 0.0514, 0.0602, 0.0585, 0.0615, 0.0556, 0.0674, 0.0703, 0.0422, 0.0850, 0.0486] # as of 10/27/2025

# Data from YFinance for all holdings (15 MIC Holdings)
portfolio_tickers = ["AMD", "AXP", "COST", "CPNG", "DUK", "EHC", "GE", "GEHC", "PM", "QCOM", "SPGI", "TMUS", "UNH", "WCN", "XYL"] # in order of weights
start = "2023-12-26"    # Start date - NOTE: Change as needed but notice that some equities have only gone public in the last 3-5 years
daily_prices = yf.download(portfolio_tickers, start=start, interval= "1d")
closing_prices = daily_prices['Close']

# Risk-Free closing prices
fed_funds_rate = 0.03625
rf_returns = fed_funds_rate / trading_days      # Daily Rf Rate

# Portfolio Returns
port_returns = closing_prices.pct_change().dropna()             # daily portfolio returns
final_port_returns = port_returns.mul(holding_weights,axis=1)   # Multipy by the Weights
final_port_returns = final_port_returns.sum(axis = 1)           # Creates a data series with the portfolio returns

#Excess Returns
excess = final_port_returns - rf_returns    # calculate the excess returns for the time period                   
excess = excess.to_frame()  # Changes the series to a dataframe

# Sharpe Ratio
sr = (excess.mean() / excess.std()) * math.sqrt(trading_days)   # Sharpe Ratio Formula: [Rp - Rf / std(Rp - Rf) * sqrt(252)]
sr = sr.iloc[0]     # isloate only the first column and row
sr = f'{sr:.3f}'    # format print the Sharpe Ratio

# Print the Sharpe Ratio and Display it
fig, ax = plt.subplots()
ax.text(0.5,0.5,str(sr),fontsize = 45, ha = "center", va = "center", color="blue")
ax.axis("Off")

# Show the Sharpe Ratio
plt.show()







