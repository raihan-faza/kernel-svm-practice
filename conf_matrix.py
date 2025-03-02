import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from train import y_pred, y_val

# Create and display the confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
plt.show()
