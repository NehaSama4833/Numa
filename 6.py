import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load
X, y = load_breast_cancer(return_X_y=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Fit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1       :", round(f1_score(y_test, y_pred), 4))

# 7. Plot
ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot()
plt.show()
