import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# loading dataset
df = pd.read_csv("pulsar_data_train.csv")
"""
# checking column info
print(df.info())

# checking the shape of the data
print(df.shape)

# preview some of the data
print(df.head())

# checking null values in percentage to decide on how to handle it
print(str(df.isna().sum() / len(df) * 100))
"""
# there is some missing value, lets check what kind of missing value we are currently facing
## checking for missing completely at random (mcar)

original_missing = df.isna().sum()

# Shuffle the dataset (randomize row order)
df_shuffled = df.sample(frac=1, random_state=42)

# Count missing values again
shuffled_missing = df_shuffled.isna().sum()

"""
# Compare missing values before and after shuffling
print("Original missing values:\n", original_missing)
print("Shuffled missing values:\n", shuffled_missing)
"""

## checking missing at random (MAR)
missing_df = df.isna().astype(int)

"""
correlations = missing_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlations, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Missing Values")
plt.show()
"""

## checking missing not at random
df["Excess_kurtosis_missing"] = (
    df[" Excess kurtosis of the integrated profile"].isna().astype(int)
)
df["Std_dev_DM_SNR_missing"] = (
    df[" Standard deviation of the DM-SNR curve"].isna().astype(int)
)
df["Skewness_DM_SNR_missing"] = df[" Skewness of the DM-SNR curve"].isna().astype(int)

# Compare distributions of missing vs. non-missing groups
features_to_compare = [" Mean of the integrated profile", " Mean of the DM-SNR curve"]

"""
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features_to_compare, 1):
    plt.subplot(1, 2, i)
    sns.boxplot(x=df["Excess_kurtosis_missing"], y=df[feature])
    plt.title(f"Comparison of {feature} based on Excess Kurtosis Missingness")
    plt.xlabel("Excess Kurtosis Missing (0 = No, 1 = Yes)")

plt.tight_layout()
plt.show()
"""

# dropping nan value
df.dropna(inplace=True)

# Setting up target variable
X = df.drop(["target_class"], axis=1)
y = df["target_class"]

# print(X)
# print(y)

# splitting the training dataset into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = pd.DataFrame(X_train, columns=[cols])
X_val = pd.DataFrame(X_val, columns=[cols])

print(X, y)
