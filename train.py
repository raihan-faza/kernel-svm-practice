from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data_process import X_train, X_val, y_train, y_val

# svm without hyper-parameter tuning
classifier = SVC()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)

print(f"model accuracy: {accuracy_score(y_val,y_pred)}")
