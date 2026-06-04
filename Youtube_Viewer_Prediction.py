import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#DATASET
video_length= [2,4,6,8,10,12,14,16]
Number_of_viewers= [15 , 25 , 40 , 50 , 65 , 75 , 90 , 100]

#DATAFRAME
df = pd.DataFrame({
    'Video_Length' : video_length,
    'Views' : Number_of_viewers
})

#print(df)
#print(df.corr())
#print(df.head())
#print(df.head(4))
#print(df.info())
print(df.describe(percentiles=[0.28]))


#GRAPH
plt.scatter(df['Video_Length'] , df['Views'])
plt.xlabel('Video_Length')
plt.ylabel('Views')
plt.title("video Time VS Views Graph")

#plt.show()

#FEATURES AND TARGET
X=df[['Video_Length']]
y=df['Views']

print(X.head())
print(y.head())

#TRAIN TEST SPLIT
X_train , X_test , y_train , y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nx-train")
print(X_train)
print("x-test")
print(X_test)

print("\ny-train")
print(y_train)
print("\ny-test")
print(y_test)


model = LinearRegression()

model.fit(X_train, y_train)


print("Slope:", model.coef_)
print("Intercept:", model.intercept_)

prediction = model.predict([[20]])
print(prediction)