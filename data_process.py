import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd

# loading dataset
df = pd.read_csv("pulsar_data_train.csv")

# checking column info
print(df.info())

# checking the shape of the data
print(df.shape)

# preview some of the data
print(df.head())

# checking null values in percentage to decide on how to handle it
print(str(df.isna().sum() / len(df) * 100))

# there is some missing value, lets check what kind of missing value we are currently facing
## checking for missing completely at random (mcar)
