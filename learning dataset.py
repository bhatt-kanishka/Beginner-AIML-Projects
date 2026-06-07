import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\HP\Downloads\train.csv")

print(df.head())
print(df.shape)
print(df.columns)

print(df.info())

print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].median())
print(df["Age"].isnull().sum())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df["Embarked"].isnull().sum())

df.drop("Cabin" , axis=1 , inplace=True)

print(df.columns)

print(df.isnull().sum())

print(df["Embarked"].unique())

print(df["Sex"].unique())

df["Sex"] = df["Sex"].map({
    "male" : 0,
    "female" : 1
})

print(df["Sex"].head())

df["Embarked"] = df["Embarked"].map({
    "S":0,
    "C":1,
    "Q":2
    })

print(df["Embarked"].head())

y = df["Survived"]

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

print(X_train.shape)

print(X_test.shape)
print(X.columns)
print(X.shape)