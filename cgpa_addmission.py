import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model  import LogisticRegression

data={
    "cgpa" : [6.67 , 5.5 ,7.2, 7.5,7.778 , 7.67,7.65 , 7.599 , 8.0 , 8.5, 10.0],
    "admission_Chance" : [0,0,0,0,1,1,1,0,1,1,1]
}


df = pd.DataFrame(data)

X = df[["cgpa"]]
y=df["admission_Chance"]

model = LogisticRegression()

model.fit(X,y)

new_data = pd.DataFrame({
    "cgpa": [7.2]
})
prediction = model.predict(new_data)
print(prediction)

print(df)

for x in [7.2, 7.4, 7.6, 7.8]:
    test = pd.DataFrame({"cgpa":[x]})
    print(x, model.predict_proba(test))

plt.scatter(df["cgpa"] , df["admission_Chance"])
plt.xlabel("cgpa")
plt.ylabel("admission_Chance")
plt.show()