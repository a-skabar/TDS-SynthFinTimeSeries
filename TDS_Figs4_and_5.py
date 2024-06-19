import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta

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

# Get returns and 1-day delayed returns
data = yf.download(ticker, start, end)
data['Return'] = data['Adj Close'].pct_change()
data['Delayed_Return'] = data['Return'].shift(1)
data.dropna(inplace=True)

silverman_width = (4/(3*data['Return'].shape[0]))**0.2 * np.std(data['Return'])
A = 0.85
k = 14.7   # Optimized for this dataset in a separate procedure
variable_widths = A * silverman_width * np.exp(k * np.abs(data['Return']-np.mean(data['Return'])))     

y_min = -0.085
y_max = 0.085
x_range = np.linspace(y_min, y_max, 1000)
bins = 'auto'

plt.rcParams.update({'font.size': 12})

# Plot Figure 4
fig4, ax1 = plt.subplots(1, 1, figsize=(10, 6))
x = data['Return']
y = data['Delayed_Return']
# Scatter plot of the data points
ax1.scatter(x, y, color='black', s=4, label='Data points')
y_target=-0.06
ax1.axhline(y=y_target, color='red', linestyle='--', linewidth=2, label=f'y = {y_target}')
y_target=-0.03
ax1.axhline(y=y_target, color='green', linestyle='--', linewidth=2, label=f'y = {y_target}')
y_target=+0.07
ax1.axhline(y=y_target, color='blue', linestyle='--', linewidth=2, label=f'y = {y_target}')
# Plot details
ax1.set_title(r'Data points ($r_t$, $r_{t-1})$ for three-year Apple series')
ax1.set_xlabel(r'Historical % return, $r_t$')
ax1.set_ylabel(r'Historical % return, $r_{t-1}$')
ax1.set_xlim([y_min, y_max])
ax1.set_ylim([y_min, y_max])
ax1.grid(True)
fig4.tight_layout()
fig4.savefig('Fig4.png', dpi=600, bbox_inches="tight")

# Plot Figure 5
fig5, ax2 = plt.subplots(1, 1, figsize=(10, 6))
# Plot the data and pdf using variable-width kernels
ax2.hist(data['Return'], density=True, alpha=0.15, bins=bins)
ax2.set_title('Conditional PDF estimates')
ax2.set_xlabel('% Return')
ax2.set_xlim([y_min, y_max])
ax2.grid(True)

bandwidth = variable_widths  # Adjust the bandwidth as needed

# Plot a variety of pdfs
weights = np.ones_like(data['Return'])  # uniform weights
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
# ax2.plot(x_range, kde_values, color = 'black', linestyle='--', label=r'PDF of $r_t$', linewidth=2)
ax2.plot(x_range, kde_values, color = 'black', linestyle='--', label=r'$P(R_t$)', linewidth=2)
y_target = -0.06
weights = np.exp(-(  ((data['Delayed_Return'] - y_target)/np.std(data['Delayed_Return'])) ** 2))
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax2.plot(x_range, kde_values, color='red',label=r'$P(R_t\,|\,P(R_{t-1}=$'+f'{y_target}', linewidth=2)

y_target = -0.03
weights = np.exp(-(  ((data['Delayed_Return'] - y_target)/np.std(data['Delayed_Return'])) ** 2))
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax2.plot(x_range, kde_values, color='green',label=r'$P(R_t\,|\,P(R_{t-1}=$'+f'{y_target}', linewidth=2)

y_target = +0.07
weights = np.exp(-(  ((data['Delayed_Return'] - y_target)/np.std(data['Delayed_Return'])) ** 2))
kde_values = weighted_kde(data['Return'], x_range, weights, bandwidth)
ax2.plot(x_range, kde_values, color='blue',label=r'$P(R_t\,|\,P(R_{t-1}=$'+f'{y_target}', linewidth=2)

ax2.legend()

fig5.tight_layout()
fig5.savefig('Fig5.png', dpi=600, bbox_inches="tight")
