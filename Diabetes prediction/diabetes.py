import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# step 1 importing data set
dataset=pd.read_csv("diabetes.csv")
# null data
dataset.info()
# correlation plot of independent var

plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(),annot=True,fmt=".3f",cmap="YlGnBu")
plt.title('Correlation Heatmap')
#plt.show()

#exploring target var
plt.figure(figsize=(10,8))

kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1],color="Red", fill=True)
kde = sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0],color="Blue", fill=True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["positive","Negative"])
#plt.show()

#exploring Glucose
plt.figure(figsize=(10,8))

kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1],color="Red", fill=True)
kde = sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0],color="Blue", fill=True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend(["positive","Negative"])
#plt.show()


# replacing 0 values with mean/median of resp feature
#Glucose

dataset["Glucose"] = dataset["Glucose"].replace(0,dataset["Glucose"].median())
#BloodPressure

dataset["BloodPressure"] = dataset["BloodPressure"].replace(0,dataset["BloodPressure"].median())
#BMI

dataset["BMI"] = dataset["BMI"].replace(0,dataset["BMI"].median())
#SkinThickness

dataset["SkinThickness"] = dataset["SkinThickness"].replace(0,dataset["SkinThickness"].median())

#spliting dependent and independent

x= dataset.drop(["Outcome"],axis=1)
y= dataset["Outcome"]

# training and testing data

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.33,random_state=42)

#KNN model 

training_accuracy=[]
test_accuracy=[]
for n_neighbors in range(1,11):
    knn= KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
plt.plot(range(1,11), training_accuracy, marker='o', label="Training Accuracy")
plt.plot(range(1,11), test_accuracy, marker='s', label="Test Accuracy")

plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()

knn= KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train)," : training accuracy")
print(knn.score(x_test,y_test)," : test accuracy")


# decision tree

dt= DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train)," : training accuracy using decision tree ")
print(dt.score(x_test,y_test)," : test accuracy using decision tree")

#MLP

mlp=MLPClassifier(random_state=42)
mlp.fit(x_train,y_train)
print(mlp.score(x_train,y_train)," : training accuracy using MLP")
print(mlp.score(x_test,y_test)," : test accuracy using MLP")
 
 
#preprocessing

sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)
mlp1=MLPClassifier(random_state=42)
mlp1.fit(x_train_scaled,y_train)
print(mlp1.score(x_train_scaled,y_train)," : training accuracy after preprocessing")
print(mlp1.score(x_test_scaled,y_test)," : test accuracy after preprocessing")

