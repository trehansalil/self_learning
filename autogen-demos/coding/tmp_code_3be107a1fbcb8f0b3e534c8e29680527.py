from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock symbols and the date range
tickers = ['AMZN', 'META', 'AAPL']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # Last 12 months

# Fetch the stock data
data = yf.download(tickers, start=start_date, end=end_date)

# Check the column names to ensure 'Adj Close' is present
print(data.columns)

# Extract the adjusted closing prices
if 'Adj Close' in data.columns:
    adj_close_data = data['Adj Close']
else:
    print("Adj Close column not found. Using Close column instead.")
    adj_close_data = data['Close']

# Plot the stock prices
plt.figure(figsize=(14, 7))
for ticker in tickers:
    plt.plot(adj_close_data.index, adj_close_data[ticker], label=ticker)

plt.title('Stock Prices of AMZN, META, and AAPL Over the Last 12 Months')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.grid(True)
plt.show()
