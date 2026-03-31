import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime
import pandas_datareader.data as web
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Defining static portfolio weights for MIC portfolio
weights = {
    "AMD" : 0.0959,
    "AXP" : 0.0939,
    "COST" : 0.1400,
    "CPNG" : 0.0320,
    "DUK" : 0.0555,
    "EHC" : 0.0515,
    "GE" : 0.0572,
    "GEHC" : 0.0528,
    "PM" : 0.0694,
    "QCOM" : 0.0445,
    "SPGI" : 0.0581,
    "TMUS" : 0.0829,
    "UNH" : 0.0365,
    "WCN" : 0.0849,
    "XYL" : 0.0449,
}

tickers = list(weights.keys())
weights = pd.Series(weights)

assert abs(weights.sum() - 1) < 1e-6, "Weights must sum to 1"

# Download historical price data for the past 2 years
end_date = datetime.today()
start_date = end_date - pd.DateOffset(years=2)

prices = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1wk",
    auto_adjust=True,
    progress=False,
)["Close"]

# Drop empty rows
prices = prices.dropna(how="all")

# Calculate weekly returns
asset_returns = prices.pct_change().dropna()

# Load and clean up Fama-French three-factor data
ff5_data = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start_date, end=end_date)
ff5_daily = ff5_data[0]/100
ff5_daily.index = pd.to_datetime(ff5_daily.index, format="%Y%m%d")

print(ff5_daily.head())
print(ff5_daily.tail())

# Align datasets to same weekly period before merging
ff5_weekly = (1 + ff5_daily).resample("W-FRI").prod() - 1
asset_returns.index = asset_returns.index.to_period("W")
ff5_weekly.index = ff5_weekly.index.to_period("W")

# Merge asset returns with Fama-French data
data=pd.concat([asset_returns, ff5_weekly], axis=1, join="inner")

# Initialize factor matrix
factors = data[["Mkt-RF", "SMB", "RMW", "CMA"]]
betas=pd.DataFrame(index=tickers, columns=factors.columns, dtype=float)
alphas=pd.Series(index=tickers, dtype=float)
residual_variance=pd.Series(index=tickers, dtype=float)
r_squared=pd.Series(index=tickers, dtype=float)

# Run regression for each stock
for ticker in tickers:
    y = data[ticker] - data["RF"]
    X = sm.add_constant(factors)

    model = sm.OLS(y, X).fit()
    #print(f"Regression results for {ticker}:")
    #print(model.summary())

    alphas[ticker] = model.params["const"]
    betas.loc[ticker] = model.params[["Mkt-RF", "SMB", "RMW", "CMA"]].values
    residual_variance[ticker] = np.var(model.resid, ddof=1)
    r_squared[ticker] = model.rsquared

# Display exposure matrix
B=betas.astype(float)
print("Factor Exposure Matrix (B):")
print(B)

# Calculate total portfolio factor exposure vector
portfolio_exposure = weights @ B
print("\nPortfolio Factor Exposure:")
print(portfolio_exposure)

# Calculate factor covariance and correlation matrix
sigma_f = factors.cov()
print("\nFactor Covariance Matrix:")
print(sigma_f)
print("\nFactor Correlation Matrix:")
print(sigma_f.corr())

# Calculate asset covariance and correlation matrices using the factor model
D=np.diag(residual_variance.values)
B_matrix = B.values
sigma_f_matrix = sigma_f.values

asset_covariance = B_matrix @ sigma_f_matrix @ B_matrix.T + D
asset_covariance_df = pd.DataFrame(asset_covariance, index=tickers, columns=tickers)
print("\nAsset Covariance Matrix:")
print(asset_covariance_df)

asset_correlation = asset_covariance_df.corr()
print("\nAsset Correlation Matrix:")
print(asset_correlation)

# Visualize factor exposure heatmap
plt.figure(figsize=(8,10))
sns.heatmap(B, annot=True, center=0, cmap="coolwarm", square=True)
plt.title("Factor Exposure Heatmap")
plt.xlabel("Factors")
plt.ylabel("Assets")
plt.show()

# Calculate portfolio variance and volatility
w=weights.values.reshape(-1,1)
portfolio_variance = w.T @ asset_covariance @ w
portfolio_volatility = np.sqrt(portfolio_variance)
print("\nPortfolio Volatility:", float(portfolio_volatility))

# Risk Decomposition
factor_risk = w.T @ B_matrix @ sigma_f_matrix @ B_matrix.T @ w
idiosyncratic_risk = w.T @ D @ w
print("\nFactor Risk Contribution:", float(np.sqrt(factor_risk)))
print("Idiosyncratic Risk Contribution:", float(np.sqrt(idiosyncratic_risk)))

# Factor Risk Contributions
portfolio_factor_exposure = weights @ B_matrix
factor_contributions = portfolio_factor_exposure * (sigma_f_matrix @ portfolio_factor_exposure)

factor_contributions = pd.Series(factor_contributions, index=factors.columns)
print("\nFactor Risk Contributions:")
print(factor_contributions)

# Regression Diagnostics
print("\nAlpha Estimates (weekly):")
print(alphas)
print("\nR-squared Values:")
print(r_squared)

# Visualize correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(asset_correlation, annot=True, center=0, cmap="coolwarm", square=True)
plt.title("Asset Correlation Matrix")
plt.show()

##############################################################

# Pull data for new stock
new_ticker = "PYPL"
new_price = yf.download(new_ticker, start=start_date, end=end_date, interval="1wk", auto_adjust=True, progress=False)["Close"]
new_returns = new_price.pct_change().dropna()
new_returns.index = new_returns.index.to_period("W")

# Merge new stock returns with Fama-French data and calculate betas
new_data = pd.concat([new_returns, data[["RF", "Mkt-RF", "SMB", "RMW", "CMA"]]], axis=1, join="inner")
y = new_data[new_ticker] - new_data["RF"]
X = sm.add_constant(new_data[["Mkt-RF", "SMB", "RMW", "CMA"]])
model = sm.OLS(y, X).fit()
new_beta = model.params[["Mkt-RF", "SMB", "RMW", "CMA"]].values
print(f"\nNew stock {new_ticker} factor exposures:")
print(new_beta)

# Adjust portfolio weights to include new stock and recalculate portfolio factor exposure
new_weight = 0.0625
adjusted_weights = weights * (1 - new_weight)
adjusted_weights[new_ticker] = new_weight
extended_betas = betas.copy()
extended_betas.loc[new_ticker] = new_beta

# Compare original and adjusted portfolio exposures
adjusted_portfolio_exposure = adjusted_weights @ extended_betas
print("\nAdjusted Portfolio Factor Exposure with new stock:")
print(adjusted_portfolio_exposure)

exposure_change = adjusted_portfolio_exposure - portfolio_exposure
comparison = pd.DataFrame({
    "Original Exposure": portfolio_exposure,
    "Adjusted Exposure": adjusted_portfolio_exposure,
    "Change": exposure_change
})
print("\nPortfolio Exposure Comparison:")
print(comparison)