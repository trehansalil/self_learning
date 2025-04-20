from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import yfinance as yf

# Define the ticker symbols
tickers = ['META', 'AAPL']

# Calculate the date for 12 months ago from today
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Fetch the stock data
data = yf.download(tickers, start=start_date, end=end_date)

# Extract the 'Close' prices for each ticker
close_data = data['Close']

# Plot the stock prices
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(close_data[ticker], label=ticker)

plt.title('Stock Prices of META and AAPL (Last 12 Months)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
