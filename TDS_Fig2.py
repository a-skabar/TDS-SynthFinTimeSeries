import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define a custom weighted KDE function
def weighted_kde(x, x_points, weights, bandwidth):
    kde_values = np.zeros_like(x_points)
    for i in range(len(x_points)):
        kde_values[i] = np.sum(weights * norm.pdf(x_points[i], x, bandwidth)) / np.sum(weights)
    return kde_values

# Get historical data for Apple (AAPL)
ticker = 'AAPL'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()

# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1

silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
k = 14.7   # Optimized in separate procedure
k = 15
variable_widths = 0.85 * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     

y_min = -0.1
y_max = 0.1
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'

# Set global fon size
plt.rcParams.update({'font.size': 12})  # Change 14 to your desired font size

# Plot figures
fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(20, 5.5), sharey=False)

# Plot the data and pdf using overly small width kernels
bandwidth = 0.0015
ax1.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax1.set_title(f'KDE using bandwidth {bandwidth}')
ax1.set_xlabel('% Return')
ax1.set_xlim([y_min, y_max])
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax1.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

# Plot the data and pdf using overly large width kernels
bandwidth = 0.009
ax2.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax2.set_title(f'KDE using bandwidth {bandwidth}')
ax2.set_xlabel('% Return')
ax2.set_xlim([y_min, y_max])
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax2.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

# Plot the data and pdf using Silvermen width kernels
bandwidth = round(silverman_width, 5)
ax3.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax3.set_title(f'KDE using {bandwidth} (Silverman bandwidth)')
ax3.set_xlabel('% Return')
ax3.set_xlim([y_min, y_max])
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax3.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

fig.tight_layout()
fig.savefig('Fig2.png', dpi=600, bbox_inches="tight")
