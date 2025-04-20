import os

import matplotlib.pyplot as plt
import yfinance as yf

# Define the ticker symbol and the period
ticker = 'META'
period = '1y'

# Fetch the historical stock price data
data = yf.download(ticker, period=period)

# Calculate the 50-day and 200-day Moving Averages (DMA)
data['50_DMA'] = data['Close'].rolling(window=50).mean()
data['200_DMA'] = data['Close'].rolling(window=200).mean()

# Calculate the MACD
data['12_EMA'] = data['Close'].ewm(span=12, adjust=False).mean()
data['26_EMA'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['12_EMA'] - data['26_EMA']
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Calculate the Relative Strength Index (RSI)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# Create a directory to save the plots
plot_dir = 'META_plots'
os.makedirs(plot_dir, exist_ok=True)

# Plot the stock price along with the 50 DMA and 200 DMA
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['50_DMA'], label='50 DMA')
plt.plot(data['200_DMA'], label='200 DMA')
plt.title('META Stock Price with 50 DMA and 200 DMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'META_price_with_DMA.png'))
plt.show()

# Plot the MACD and Signal Line
plt.figure(figsize=(14, 7))
plt.plot(data['MACD'], label='MACD')
plt.plot(data['Signal'], label='Signal Line')
plt.title('META MACD and Signal Line')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'META_MACD.png'))
plt.show()

# Plot the RSI
plt.figure(figsize=(14, 7))
plt.plot(data['RSI'], label='RSI')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('META RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.savefig(os.path.join(plot_dir, 'META_RSI.png'))
plt.show()

# Save the data to a CSV file
data.to_csv(os.path.join(plot_dir, 'META_technical_analysis.csv'))
