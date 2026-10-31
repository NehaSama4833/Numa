import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("titanic.csv")
print(df.isnull().sum())                       # 1. find missing values

df = df.drop(columns=["Cabin", "PassengerId", "Name", "Ticket"])   # 2. drop useless columns
df["Age"] = df["Age"].fillna(df["Age"].median())                   # fill numbers with median
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])   # fill categories with mode

df["Sex"] = LabelEncoder().fit_transform(df["Sex"])                # 3. text -> 0/1
df = pd.get_dummies(df, columns=["Embarked"], dtype=int)           # one column per port

df[["Age", "Fare"]] = StandardScaler().fit_transform(df[["Age", "Fare"]])   # 4. scale
print(df.shape)
