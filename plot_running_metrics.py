"""Script to plot running metrics"""
# Re-import required libraries after code execution reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load the most recent spreadsheet
latest_file_path = "fitness_tracker.csv"
df_updated = pd.read_csv(latest_file_path)

# Fix and convert date formatting
df_updated['Date'] = df_updated['Date'].apply(lambda x: f"{x}-2025" if '-' in x and len(x.split('-')[1]) == 3 else x)
df_updated['Date'] = pd.to_datetime(df_updated['Date'])
df_updated = df_updated.sort_values(by='Date')

# Convert dates to ordinal for regression
date_nums_updated = df_updated['Date'].map(pd.Timestamp.toordinal)

# Rolling slope calculation (10-run window)
window_size = 12
R2 = []

if len(df_updated) >= window_size:
    for i in range(len(df_updated) - window_size + 1):
        x = date_nums_updated.iloc[i:i + window_size].values
        y = df_updated['Average Pace (min/mile)'].iloc[i:i + window_size].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        R2.append(r_value ** 2.)

    slope_dates = df_updated['Date'].iloc[window_size - 1:].reset_index(drop=True)
else:
    R2 = []
    slope_dates = []

# Linear regression for average pace
slope_updated, intercept_updated, *_ = stats.linregress(date_nums_updated, df_updated['Average Pace (min/mile)'])

# Predict date for 7.8 min/mile
target_pace = 7.8
target_date_num_updated = (target_pace - intercept_updated) / slope_updated
predicted_date_updated = pd.Timestamp.fromordinal(int(target_date_num_updated))

# === Plot 1: Projected Pace Over Time ===
plt.figure(figsize=(8, 5))
plt.plot(df_updated['Date'], df_updated['Average Pace (min/mile)'], color="orange", marker='o', linestyle='-', label='Data')

# Trendline
x_vals = np.linspace(min(date_nums_updated), max(date_nums_updated) + 200, 200)
y_vals = slope_updated * x_vals + intercept_updated
dates_fit = [pd.Timestamp.fromordinal(int(d)) for d in x_vals]
plt.plot(dates_fit, y_vals, linestyle='--', color='blue', label='Linear Trendline')

# Projection markers
plt.axhline(y=7.8, color='gray', linestyle=':', label='Target Pace (7.8 min/mile)')
plt.axvline(x=predicted_date_updated, color='red', linestyle='--', label=f'Predicted Date\n({predicted_date_updated.date()})')
plt.scatter([predicted_date_updated], [7.8], color='red', marker="*", zorder=5)

plt.title('Projected Average Pace Over Time')
plt.xlabel('Date')
plt.ylabel('Average Pace (min/mile)')
plt.grid(True)
plt.legend()
plt.gca().minorticks_on()
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("average_pace_projection.png", format="png")

# === Plot 3: 3-panel chart with R² and p-values included ===
plt.figure(figsize=(15, 14))

# Helper to plot each subplot with regression stats
def plot_metric_with_stats(x, y, index, title, ylabel):
    plt.subplot(4, 1, index)
    plt.plot(df_updated['Date'], y, marker='o', color="orange", linestyle='-', label='Data')

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = slope * x_vals + intercept
    dates_fit = [pd.Timestamp.fromordinal(int(d)) for d in x_vals]
    plt.plot(dates_fit, y_vals, linestyle='--', color='blue', label='Linear Trendline')

    r_squared = r_value ** 2
    stats_text = f"$R^2$ = {r_squared:.2f}, p = {p_value:.3g}"
    plt.text(0.01, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.gca().minorticks_on()
    plt.xticks(rotation=0)

# Plot each metric
plot_metric_with_stats(date_nums_updated, df_updated['Average Pace (min/mile)'], 1,
                       'Date vs. Average Pace (min/mile)', 'Average Pace (min/mile)')

plot_metric_with_stats(date_nums_updated, df_updated['Average Heart Rate (beats/min)'], 2,
                       'Date vs. Average Heart Rate (beats/min)', 'Average Heart Rate (bpm)')

plot_metric_with_stats(date_nums_updated, df_updated['Beats per Mile'], 3,
                       'Date vs. Beats per Mile', 'Beats per Mile')

plt.subplot(4, 1, 4)
plt.plot(slope_dates, R2, marker='o', linestyle='-', color='purple')
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2
stats_text = f"$R^2$ = {r_squared:.2f}, p = {p_value:.3g}"
plt.text(0.01, 0.95, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
plt.title(r'Change in R$^{2}$ over Time (%s Day Rolling Measurement)' % str(window_size))
plt.xlabel('Date')
plt.ylabel(r'R$^{2}$')
plt.grid(True)
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("three_panel_metrics.png", format="png")
