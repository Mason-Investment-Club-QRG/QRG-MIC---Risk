# Project (1b) - Quantitative Risk Group (Risk Analyst: Abishek Samuel)
# Portfolio Beta Calculation using Regression Analysis
# Beta - average of 2y weekly data
# Graph and "Beta Line" displayed

import pandas as pd
import yfinance as yf
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Portfolio Holding and Weights

port_holdings = ["AMD", "AXP", "COST", "CPNG", "DUK", "EHC", "GE", "GEHC", "PM", "QCOM", "SPGI", "TMUS", "UNH", "WCN", "XYL"] # as of 10/27/2025
port_weights = [0.0958, 0.1091, 0.1111, 0.0386, 0.0447, 0.0514, 0.0602, 0.0585, 0.0615, 0.0556, 0.0674, 0.0703, 0.0422, 0.0850, 0.0486] # in order of tickers

# Import Financial Data (Closing Prices for the portfolio and SPY)
start = "2023-12-29" # NOTE - Must Update as Desired for Results
end = "2025-12-29" # NOTE - Must Update as Desired for Results
holdings_prices = yf.download(port_holdings, start=start, end=end, interval="1wk", auto_adjust=True, actions=True)["Close"].dropna() #Yahoo Finance
spy_ticker = "SPY"  # Benchmark Ticker
spy_prices = yf.download(spy_ticker, start=start, end=end, interval="1wk", auto_adjust=True, actions=True)["Close"].dropna() #Yahoo Finance

# Returns (Percent Change Calculation)
holdings_returns = holdings_prices.pct_change().dropna() # Calculate holdings returns using pandas dataframe
spy_returns = spy_prices.pct_change().dropna()    # Calculate benchmark (SPY) returns using pandas dataframe

# Portfolio Returns
weights = np.array(port_weights)    # change the weights to a array using numpy dataframe                   
port_returns = holdings_returns.mul(weights,axis=1) # multiply the holdings' returns by the weights (array)
port_returns = port_returns.sum(axis=1) # sum the multiplied returns to get the final portfolio returns (series)
spy_returns = spy_returns.sum(axis=1)   # sum the benchmark (SPY) returns to get the final returns (series)

# 2-Y Weekly Beta
beta = port_returns.cov(spy_returns) / spy_returns.var()    # Staistical Beta formula (Cov(port,market) / var(market))
print(f'{beta:.2f}') # Beta of the Portfolio (Based on 2-Y Weekly data, and the use of a statitical Linear Regression Model)

# Turning series into a dataframe
x_values = port_returns.to_frame() * 100    # converts the portfolio returns into a dataframe
y_values = spy_returns.to_frame() * 100     # converts the benchmark (SPY) returns into a dataframe

# Turn into arrays
x = np.array(x_values)  #converts portfolio returns from a dataframe into an array
y = np.array(y_values)  #converts holdings returns from a dataframe into an array

# Beta Visuals (NOTE: Created with the help of ChatGPT (AI))
df = pd.concat([port_returns, spy_returns], axis=1).dropna()    # alining the existing data
df.columns = ["port", "spy"]
beta_stat = df["port"].cov(df["spy"]) / df["spy"].var() # Statistical beta - Computed again
alpha_stat = df["port"].mean() - beta_stat * df["spy"].mean() # Matching intercept: alpha = ȳ − βx̄
x = df["spy"].values * 100      
y = df["port"].values * 100     
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.scatter(x, y, alpha=0.70, color="#0A3D0A")
xx = np.linspace(-6, 6, 200)
yy = (beta_stat * (xx / 100) + alpha_stat) * 100
ax.plot(xx, yy, color="black", linewidth=4)
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.set_xlim(-6.5, 6.5)
ax.set_ylim(-6.5, 6.5)
ax.set_xticks(np.arange(-6, 7, 1))
ax.set_yticks(np.arange(-6, 7, 1))
ax.text(0.02, 0.95,f"Statistical β = {beta_stat:.2f}",transform=ax.transAxes,bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax.grid(True, linestyle="--", alpha=0.25, color="black")
plt.title("Portfolio Returns vs Benchmark Returns (Statistical Beta - 2y Weekly)")

# Show the graph
plt.show()



