import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Load and clean
df = pd.read_csv("auto_mpg.csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower"])
print("Rows:", len(df))

# 2. Split
X = df[["horsepower"]]
y = df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Try different degrees
degrees = [1, 2, 3, 5, 8, 10]
train_err, test_err = [], []
for d in degrees:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    train_err.append(mean_squared_error(y_train, model.predict(X_train)))
    test_err.append(mean_squared_error(y_test, model.predict(X_test)))
    print("Degree", d, "| Train MSE:", round(train_err[-1], 2), "| Test MSE:", round(test_err[-1], 2))
print("Best degree:", degrees[int(np.argmin(test_err))])

# 4. Plots
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(degrees, train_err, marker="o", label="Train")
ax[0].plot(degrees, test_err, marker="o", label="Test")
ax[0].set_xlabel("Degree"); ax[0].set_ylabel("MSE"); ax[0].legend(); ax[0].set_title("Error vs Degree")

x_line = pd.DataFrame({"horsepower": np.linspace(X["horsepower"].min(), X["horsepower"].max(), 200)})
ax[1].scatter(X_test, y_test, s=10, color="gray")
for d in [1, 2, 8]:
    m = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d), LinearRegression()).fit(X_train, y_train)
    ax[1].plot(x_line, m.predict(x_line), label="degree " + str(d))
ax[1].set_xlabel("horsepower"); ax[1].set_ylabel("mpg"); ax[1].legend(); ax[1].set_title("Fits")
plt.show()
