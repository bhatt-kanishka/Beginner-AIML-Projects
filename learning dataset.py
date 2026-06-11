import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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
print(df.columns)


y = df["Survived"]

X = df[
        [
            "Pclass",
            "Sex",
            "Age",
            "Parch",
            "Fare",
            "Embarked"
    ]
]


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)
model.fit(X_train , y_train)
y_pred = model.predict(X_test)


cm = confusion_matrix(y_test , y_pred)
print(cm)

knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("KNN Accuracy :", accuracy_score(y_test, y_pred_knn))
print("KNN Precision :", precision_score(y_test, y_pred_knn))
print("KNN Recall :", recall_score(y_test, y_pred_knn))
print("KNN F1 Score :", f1_score(y_test, y_pred_knn))

nb = GaussianNB()
nb.fit(X_train , y_train)
y_pred_nb = nb.predict(X_test)

print("\nNB Accuraccy : ",accuracy_score(y_test , y_pred_nb))
print("NB presecion : ",precision_score(y_test , y_pred_nb))
print("NB recall : ",recall_score(y_test , y_pred_nb))
print("NB F1 score : ",f1_score(y_test , y_pred_nb))