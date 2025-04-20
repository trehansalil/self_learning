import os

import matplotlib.pyplot as plt
import yfinance as yf

# Define the stock symbols
symbols = ['AMZN', 'META', 'AAPL']

# Fetch the stock data for the last 12 months
data = yf.download(symbols, period='1y')

# Plotting the stock prices
plt.figure(figsize=(14, 7))
for symbol in symbols:
    plt.plot(data['Close'][symbol], label=symbol)

plt.title('Stock Prices for Amazon, Meta, and Apple (Last 12 Months)')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)

# Save the plot to a folder
output_folder = 'stock_plots'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'stock_prices.png')
plt.savefig(output_path)

print(f'Plot saved to {output_path}')
plt.show()
