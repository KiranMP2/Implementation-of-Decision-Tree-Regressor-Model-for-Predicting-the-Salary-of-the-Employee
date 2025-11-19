# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.Calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KIRAN MP
RegisterNumber: 212224230123
import pandas as pd
df=pd.read_csv('salary.csv')
df.head()
df.info()
print(' ')

df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()
df["Salary"].value_counts()
x=df[["Position","Level"]]
x.head()
y=df["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
print("NAME : KIRAN MP")
print("REG NO : 212224230123")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae=mean_absolute_error(y_test,y_pred)
mae
mse=mean_squared_error(y_test,y_pred)
mse
r_square=r2_score(y_test,y_pred)
r_square
import numpy as np
rmse=np.sqrt(mse)
rmse
dt.predict([[5,6]])
 
*/
```
## Output:
df.head():

<img width="281" height="192" alt="513646282-85824529-606c-4d7d-8294-6c797af4f2be" src="https://github.com/user-attachments/assets/cd39120d-9dcb-407d-9798-2ab5f474af9b" />

df.info() and null:

<img width="399" height="305" alt="513646447-dd3778a9-9339-41fc-9dba-6ee68945ff47" src="https://github.com/user-attachments/assets/cb5c2eeb-45c0-4600-bea3-0d2e562bb06c" />

df.head():

<img width="207" height="182" alt="513646073-3c1d2dba-48b5-4827-9584-c7738048f2ff" src="https://github.com/user-attachments/assets/6ee3257c-a517-4acf-9961-092be7646854" />

value_counts():

<img width="246" height="222" alt="513646570-28749562-8970-4004-b4d4-18c753689397" src="https://github.com/user-attachments/assets/ee2f9006-86dc-4b68-93a2-30cfe6a3e8ea" />

x.head():

<img width="181" height="180" alt="513646733-6b9a4298-8103-4c70-8a5a-8fa004824a31" src="https://github.com/user-attachments/assets/edf1286b-d2c9-4f95-b67a-2fa6698d09ce" />

dt.fit(x_train,y_train):


<img width="266" height="111" alt="Screenshot 2025-11-19 210451" src="https://github.com/user-attachments/assets/af79c7c4-0613-4e4b-8e8a-30e001bbbd87" />

y_pred:

<img width="233" height="25" alt="513647108-f2d629e6-a05f-4b59-9a4f-90cd8b1672de" src="https://github.com/user-attachments/assets/1eb0eab5-d33e-4013-9c4e-c08933475dfc" />

mae:

<img width="90" height="37" alt="513647281-94017015-bb51-4d2d-9d41-bc874880506b" src="https://github.com/user-attachments/assets/494c65b2-45d7-4646-be17-771a414fc6e9" />

mse:

<img width="152" height="32" alt="513647389-78570406-7024-47c2-b728-a63a11f5b58a" src="https://github.com/user-attachments/assets/02d6f9f9-59b4-4127-ba0e-3e9f8768e1d5" />

r_square:

<img width="92" height="30" alt="513647622-9a66d67d-8bc9-48e0-8b5e-369a921a0885" src="https://github.com/user-attachments/assets/927f50aa-fbbb-431a-9072-558e07244ce1" />

rmse:

<img width="176" height="32" alt="513647766-1def14f5-cf29-4339-b833-227a511241ab" src="https://github.com/user-attachments/assets/038979ec-0b8a-4066-88ac-0350064f475e" />

predict:

<img width="1128" height="108" alt="513647852-871d4a05-080e-419d-9e6d-730e69d5881e" src="https://github.com/user-attachments/assets/18b88cd5-1a77-48d7-8a28-ac65d3d6a2c8" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
