import pandas as pd

# Load dataset
df = pd.read_csv("sensor_data1.csv")

# 1. See first 5 records
print(df.head())

# 2. Check available columns
print(df.columns)

# 3. Check data types and missing values
print(df.info())

# 4. Check statistical information
print(df.describe())

# 5. Count missing values
print(df.isnull().sum())

# 6. Count duplicate records
print("Duplicates:", df.duplicated().sum())
