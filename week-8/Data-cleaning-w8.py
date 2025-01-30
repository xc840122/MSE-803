import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("week-8/House_Data.csv")

# 1. Cleaning Null Values
# Check for null values in each column
print("Null values before cleaning:")
print(df.isnull().sum())

# Fill null values with appropriate methods
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Check for null values after cleaning
print("\nNull values after cleaning:")
print(df.isnull().sum())

# 2. Removing Duplicates
# Check for duplicated rows
print("\nNumber of duplicated rows before cleaning:", df.duplicated().sum())

# Remove duplicated rows
df = df.drop_duplicates()

# Check for duplicated rows after cleaning
print("Number of duplicated rows after cleaning:", df.duplicated().sum())

# 3. Cleaning Outliers
# Function to detect outliers using IQR
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

# Handle outliers for numerical columns
for col in numerical_cols:
    outliers = detect_outliers(df[col])
    if not outliers.empty:
        print(f"\nOutliers detected in {col}:")
        print(outliers)
        # Cap the outliers to the upper and lower bounds
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Save the cleaned data to a new CSV file
df.to_csv('week-8/Cleaned_House_Data.csv', index=False)

print("\nData cleaning completed. Cleaned data saved to 'Cleaned_House_Data.csv'.")