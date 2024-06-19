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

# Set global font size
plt.rcParams.update({'font.size': 12})  

# Plot figures
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 12), sharey=False)

# Plot variable-width pdf for AAPL
ticker = 'AAPL'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
print(ticker)
print('max: ', np.max(data['Return']))
print('min: ', np.min(data['Return']))
# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1
silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
A = 0.85
k = 14.69
variable_widths = A * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     
y_min = -0.1
y_max = 0.1
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'
ax1.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax1.set_title(f'Variable-width KDE estimate for Apple AAPL)')
ax1.set_xlabel('% Return')
ax1.set_xlim([y_min, y_max])
bandwidth = variable_widths  # Adjust the bandwidth as needed
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax1.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

# Plot variable-width pdf for GOOG
ticker = 'GOOG'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
print(ticker)
print('max: ', np.max(data['Return']))
print('min: ', np.min(data['Return']))
# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1
silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
A = 0.85
k = 20.67
variable_widths = A * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     
y_min = -0.1
y_max = 0.1
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'
ax2.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax2.set_title(f'Variable-width KDE estimate for Google (GOOG)')
ax2.set_xlabel('% Return')
ax2.set_xlim([y_min, y_max])
bandwidth = variable_widths  # Adjust the bandwidth as needed
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax2.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

# Plot variable-width pdf for META
ticker = 'META'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
print(ticker)
print('max: ', np.max(data['Return']))
print('min: ', np.min(data['Return']))
# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1
silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
A = 0.85
k = 9.11
variable_widths = A * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     
y_min = -0.3
y_max = 0.3
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'
ax3.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax3.set_title(f'Variable-width KDE estimate for Facebook (META)')
ax3.set_xlabel('% Return')
ax3.set_xlim([y_min, y_max])
bandwidth = variable_widths  # Adjust the bandwidth as needed
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax3.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

# Plot variable-width pdf for MSFT
ticker = 'MSFT'
start = '2021-01-01'
end = '2024-01-01'
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
print(ticker)
print('max: ', np.max(data['Return']))
print('min: ', np.min(data['Return']))
# Reset data index from date to day number
data = data.dropna().reset_index(drop=True)
data['Index'] = data.index + 1
silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
A = 0.85
k = 21.09
variable_widths = A * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     
y_min = -0.1
y_max = 0.1
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'
ax4.hist(data['Return'], density=True, alpha=0.5, bins=bins)
ax4.set_title(f'Variable-width KDE estimate for Microsoft (MSFT)')
ax4.set_xlabel('% Return')
ax4.set_xlim([y_min, y_max])
bandwidth = variable_widths  # Adjust the bandwidth as needed
weights = np.ones_like(data['Return'])
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax4.plot(x_range, kde_values, color='red', label='Blah Blah', linewidth=2)

fig.tight_layout()
fig.savefig('Fig3.png', dpi=600, bbox_inches="tight")
