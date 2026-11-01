import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# ----- Part A: Breast cancer, 3 kernels -----
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

for kernel in ["linear", "rbf", "poly"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(kernel, "accuracy:", round(accuracy_score(y_test, y_pred), 4))

# ----- Part B: Moons (curved data), plot the boundary -----
Xm, ym = make_moons(n_samples=200, noise=0.15, random_state=42)
Xm_train, Xm_test, ym_train, ym_test = train_test_split(Xm, ym, test_size=0.2, random_state=42, stratify=ym)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, kernel in zip(ax, ["linear", "rbf"]):
    model = SVC(kernel=kernel).fit(Xm_train, ym_train)
    print(kernel, "moons accuracy:", round(accuracy_score(ym_test, model.predict(Xm_test)), 3))
    DecisionBoundaryDisplay.from_estimator(model, Xm, ax=a, alpha=0.3)
    a.scatter(Xm[:, 0], Xm[:, 1], c=ym)
    a.set_title(kernel)
plt.show()
