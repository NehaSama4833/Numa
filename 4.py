import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

df = pd.read_csv("auto_mpg.csv")
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df = df.dropna(subset=["horsepower"])

X = df[["horsepower"]]
y = df["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for d in [1, 2, 3, 5, 8, 10]:
    model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=d), LinearRegression())
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    test_mse = mean_squared_error(y_test, model.predict(X_test))
    print("Degree", d, "| Train MSE:", round(train_mse, 2), "| Test MSE:", round(test_mse, 2))
