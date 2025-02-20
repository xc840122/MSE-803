# Readme: Explanation of the Code and Correlation Calculation

## **Code Overview**

### 1. Dataset Loading

The code utilizes the Iris dataset from `sklearn.datasets` and converts it into a Pandas DataFrame. The dataset contains four numerical features:

- `sepal_len`
- `sepal_wid`
- `petal_len`
- `petal_wid`

Additionally, a `class` column represents the species of Iris flowers.

---

### 2. Column Naming

The columns are renamed for better clarity and usability.

---

### 3. Missing Data Analysis

- **Number of Missing Values**:  
  The `isnull().sum()` function calculates the total number of missing values in each column.

- **Mean of Missing Values**:  
  The `isnull().mean()` function computes the proportion of missing values (as a fraction of total rows).

- **Cleaning**:  
  The `dropna(how="all", inplace=True)` method removes rows where all values are missing.

---

### 4. Data Preview

The first 5 rows of the feature columns (`sepal_len`, `sepal_wid`, `petal_len`, `petal_wid`) are displayed using `iloc`.

---

## **Calculating Correlation Among Features**

To calculate the correlation between numerical features, use the Pandas `corr()` function. The correlation matrix is rounded to two decimal places for readability:

```python
correlation_matrix = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].corr().round(2)
print("Correlation Matrix:\n", correlation_matrix)
```
