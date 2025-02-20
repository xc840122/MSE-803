# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example Water Quality Data
data = {
    "Site ID": ["AV-1", "AV-2", "AV-3", "AV-1", "AV-2", "AV-3"],
    "Date": ["2023-10-25", "2023-10-25", "2023-10-25", "2023-11-15", "2023-11-15", "2023-11-15"],
    "Temperature (°C)": [15.2, 14.8, 13.6, 12.4, 11.9, 10.8],
    "Dissolved Oxygen (mg/L)": [8.5, 7.9, 6.8, 8.2, 7.7, 6.2],
    "pH": [7.8, 7.5, 7.3, 7.6, 7.4, 7.2]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Group data by Site and sort by Date for analysis
df_sorted = df.sort_values(by=['Site ID', 'Date'])

# Plot time series for each parameter
plt.figure(figsize=(12, 8))

# Subplot 1: Temperature Trends
plt.subplot(3, 1, 1)
for site, group in df_sorted.groupby('Site ID'):
    plt.plot(group['Date'], group['Temperature (°C)'], marker='o', label=f"Site {site}")
plt.title("Temperature Trends Over Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid()

# Subplot 2: Dissolved Oxygen Trends
plt.subplot(3, 1, 2)
for site, group in df_sorted.groupby('Site ID'):
    plt.plot(group['Date'], group['Dissolved Oxygen (mg/L)'], marker='o', label=f"Site {site}")
plt.title("Dissolved Oxygen Trends Over Time")
plt.ylabel("DO (mg/L)")
plt.legend()
plt.grid()

# Subplot 3: pH Trends
plt.subplot(3, 1, 3)
for site, group in df_sorted.groupby('Site ID'):
    plt.plot(group['Date'], group['pH'], marker='o', label=f"Site {site}")
plt.title("pH Trends Over Time")
plt.xlabel("Date")
plt.ylabel("pH")
plt.legend()
plt.grid()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()