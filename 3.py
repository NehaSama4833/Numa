import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load
df = pd.read_csv("california_housing.csv")
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]
print(X.shape)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Fit
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Score
mse = mean_squared_error(y_test, y_pred)
print("MSE :", round(mse, 4))
print("RMSE:", round(np.sqrt(mse), 4))
print("R2  :", round(r2_score(y_test, y_pred), 4))
print(pd.Series(model.coef_, index=X.columns).round(3).sort_values())

# 7. Plot
plt.scatter(y_test, y_pred, s=3)
plt.plot([0, 5], [0, 5], color="red")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted")
plt.show()
