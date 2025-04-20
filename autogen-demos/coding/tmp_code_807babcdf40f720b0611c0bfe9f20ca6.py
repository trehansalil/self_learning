from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import ta
import yfinance as yf

# Set the directory where you want to save the plots
output_dir = 'path/to/your/output/directory'

# Fetch the last 12 months of data for Meta Platforms Inc. (META)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
df = yf.download('META', start=start_date, end=end_date)

# Calculate the 50 DMA, 200 DMA, MACD, and RSI
df['50_DMA'] = df['Close'].rolling(window=50).mean()
df['200_DMA'] = df['Close'].rolling(window=200).mean()

# Calculate MACD
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

# Calculate RSI
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()

# Drop rows with NaN values caused by the rolling window
df = df.dropna()

# Plotting the Close price with 50 DMA and 200 DMA
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.plot(df.index, df['50_DMA'], label='50 DMA', color='orange')
plt.plot(df.index, df['200_DMA'], label='200 DMA', color='red')
plt.title('Meta Platforms Inc. (META) Close Price with 50 DMA & 200 DMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'{output_dir}/META_50_200_DMA.png')
plt.show()

# Plotting the MACD and Signal Line
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['MACD'], label='MACD', color='blue')
plt.plot(df.index, df['MACD_Signal'], label='MACD Signal', color='red')
plt.title('Meta Platforms Inc. (META) MACD & Signal Line')
plt.xlabel('Date')
plt.ylabel('MACD Values')
plt.legend()
plt.savefig(f'{output_dir}/META_MACD.png')
plt.show()

# Plotting the RSI
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['RSI'], label='RSI', color='green')
plt.axhline(30, linestyle='--', color='red', label='Overbought (30)')
plt.axhline(70, linestyle='--', color='red', label='Oversold (70)')
plt.title('Meta Platforms Inc. (META) RSI')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.savefig(f'{output_dir}/META_RSI.png')
plt.show()
