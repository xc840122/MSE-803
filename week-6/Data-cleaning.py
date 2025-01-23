
################ Data cleaning the Iris dataset #################
from sklearn import datasets
import pandas as pd

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
#### ===> TASK 1: here - add two more lines of the code to find the number and mean of missing data
missing_data_count = iris_df.isnull().sum()  # Find the number of missing values per column
missing_data_mean = iris_df.isnull().mean()  # Find the mean of missing values per column

print("Number of missing values:\n", missing_data_count)
print("Mean of missing values:\n", missing_data_mean)

cleaned_data = iris_df.dropna(how="all", inplace=True) # remove any empty lines

iris_X=iris_df.iloc[:5,[0,1,2,3]]
print(iris_X)

### TASK2: Here - Write a short readme to explain above code and how we can calculate the corrolation amoung featuers with description
correlation_matrix = iris_df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].corr().round(2)
print("Correlation Matrix:\n", correlation_matrix)