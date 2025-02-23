import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

original_missing = df.isna().sum()

# Shuffle the dataset (randomize row order)
df_shuffled = df.sample(frac=1, random_state=42)

# Count missing values again
shuffled_missing = df_shuffled.isna().sum()

# Compare missing values before and after shuffling
print("Original missing values:\n", original_missing)
print("Shuffled missing values:\n", shuffled_missing)

## checking missing at random (MAR)
missing_df = df.isna().astype(int)

correlations = missing_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Missing Values")
plt.show()
