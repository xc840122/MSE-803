import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load water quality and fish population data
water_data = pd.DataFrame({
    'Site ID': ['AV-1', 'AV-2', 'AV-3', 'AV-1', 'AV-2', 'AV-3'],
    'Date': ['2023-10-25', '2023-10-25', '2023-10-25', '2023-11-15', '2023-11-15', '2023-11-15'],
    'Temperature': [15.2, 14.8, 13.6, 12.4, 11.9, 10.8],
    'pH': [7.8, 7.5, 7.3, 7.6, 7.4, 7.2],
    'Dissolved Oxygen': [8.5, 7.9, 6.8, 8.2, 7.7, 6.2]
})

fish_data = pd.DataFrame({
    'Site ID': ['AV-1', 'AV-2', 'AV-3', 'AV-1', 'AV-2', 'AV-3'],
    'Date': ['2023-10-25', '2023-10-25', '2023-10-25', '2023-11-15', '2023-11-15', '2023-11-15'],
    'Species': ['Brown Trout', 'Shortfin Eel', 'Freshwater Shrimp', 'Rainbow Trout', 'Kokopu', 'No fish observed'],
    'Count': [50, 75, 120, 35, 40, None],
    'Average Size': [32, 45, 2, 28, 18, None]
})

# Correlation Analysis
merged_data = pd.merge(water_data, fish_data, on=['Site ID', 'Date'])
correlation_matrix = merged_data[['Temperature', 'pH', 'Dissolved Oxygen', 'Count']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Regression Analysis
X = merged_data[['Temperature', 'pH', 'Dissolved Oxygen']].dropna()
y = merged_data['Count'].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Visualization
plt.scatter(merged_data['Dissolved Oxygen'], merged_data['Count'])
plt.xlabel('Dissolved Oxygen (mg/L)')
plt.ylabel('Fish Count')
plt.title('Fish Count vs Dissolved Oxygen')
plt.show()