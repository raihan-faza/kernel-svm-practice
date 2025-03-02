import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from train import y_pred, y_val

# Generate the classification report as a dictionary
report_dict = classification_report(y_val, y_pred, output_dict=True)

# Convert to a Pandas DataFrame
df = pd.DataFrame(report_dict).transpose()

# Create a figure
plt.figure(figsize=(8, 4))
sns.heatmap(df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="Blues", cbar=False)

# Save as an image
plt.title("Classification Report")
plt.savefig("classification_report.png", dpi=300, bbox_inches="tight")
plt.show()
