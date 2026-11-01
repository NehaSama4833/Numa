import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load and check missing values
df = pd.read_csv("titanic.csv")
print(df.shape)
print(df.isnull().sum())

# 2. Handle missing values
df = df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print("Missing left:", df.isnull().sum().sum())

# 3. Encode text columns
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)

# 4. Create a new feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 5. Scale numbers
df[["Age", "Fare"]] = StandardScaler().fit_transform(df[["Age", "Fare"]])

print(df.shape)
print(df.head())
