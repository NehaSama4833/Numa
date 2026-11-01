import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load
X, y = load_diabetes(return_X_y=True, as_frame=True)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. VIF (multicollinearity check)
for i, name in enumerate(X.columns):
    others = np.delete(X_train, i, axis=1)
    r2 = LinearRegression().fit(others, X_train[:, i]).score(others, X_train[:, i])
    print(name, "VIF:", round(1 / (1 - r2), 1))

# 5. Fit and score three models
models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=10), "Lasso": Lasso(alpha=1)}
coefs = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coefs[name] = model.coef_
    print(name, "| Test MSE:", round(mean_squared_error(y_test, y_pred), 1),
          "| R2:", round(r2_score(y_test, y_pred), 4),
          "| Zero coefs:", int((model.coef_ == 0).sum()))

# 6. Plot coefficients
pd.DataFrame(coefs, index=X.columns).plot(kind="bar", figsize=(8, 4))
plt.title("Coefficients: Linear vs Ridge vs Lasso")
plt.show()
