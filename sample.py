import pandas as pd
prices = pd.read_csv("data/sample_prices.csv")
returns = prices.pct_change()
print(returns.dropna())
print(returns.std())