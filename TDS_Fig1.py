import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Get historical data for Apple (AAPL)
ticker = 'AAPL'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
data['Close'] = data['Adj Close']

# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1

bins='auto'
y_min = -0.1
y_max = 0.1

# Set global font size
plt.rcParams.update({'font.size': 12})  # Change 14 to your desired font size

# Plot figures
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 5.5),  sharey=False)

ax1.plot(data['Close'], alpha=0.5)
ax1.set_title('Daily Close Price for Apple (AAPL)')
ax1.set_xlabel('Day')

ax2.plot(data['Return'], alpha=0.5)
ax2.set_title('Daily % Returns for Apple (AAPL)')
ax2.set_xlabel('Day')
ax2.set_ylim([y_min, y_max])

# Fit a normal distribution to the data
mu, std = norm.fit(data['Return'])
x = np.linspace(y_min, y_max, 1000)
p = norm.pdf(x, mu, std)
ax3.plot(x, p, color = 'black', linestyle='--', linewidth=2)
ax3.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax3.set_title('Distribution of % Daily Returns for Apple (AAPL)')
ax3.set_xlabel('% Return')
ax3.set_xlim([y_min, y_max])

fig.tight_layout()
fig.savefig('Fig1.png', dpi=600, bbox_inches="tight")
 