# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.

2. Upload and read the dataset.

3. Check for any null values using the isnull() function.

4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Thaksha Rishi
RegisterNumber:  212223100058
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("Employee.csv")

print("Name: Thaksha Rishi")
print("Reg No: 212223100058\n")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)

# data.head()

<img width="1251" height="295" alt="image" src="https://github.com/user-attachments/assets/5abb2bb8-11c5-44ce-acb6-ac0a5a1cbbbb" />

# data.info()

<img width="541" height="366" alt="image" src="https://github.com/user-attachments/assets/1e57b950-75af-49e8-8562-f9b460401736" />

# data.isnull().sum()

<img width="404" height="258" alt="image" src="https://github.com/user-attachments/assets/12e053f2-c90e-4450-b3a1-b83b6943f2a6" />

# data.value_counts()

<img width="362" height="82" alt="image" src="https://github.com/user-attachments/assets/382fa845-3113-4c9d-9369-19360f58e46e" />

# x.head()

# accuracy

<img width="447" height="82" alt="image" src="https://github.com/user-attachments/assets/032b6803-0f82-4655-991c-1f523219ab87" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
