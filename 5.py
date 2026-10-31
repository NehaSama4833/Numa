from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=10), "Lasso": Lasso(alpha=1)}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, "| Test MSE:", round(mean_squared_error(y_test, y_pred), 1),
          "| R2:", round(r2_score(y_test, y_pred), 4),
          "| Zero coefs:", int((model.coef_ == 0).sum()))
