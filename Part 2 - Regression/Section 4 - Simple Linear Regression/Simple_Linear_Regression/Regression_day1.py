# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:26:38 2018

@author: Nirale1.Kumar
"""

#Regression Techniques...

# Data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Creating train and test data
from sklearn.cross_validation import train_test_split                

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
                
# Fittinf simple Linear regression model to our training set

from sklearn.linear_model import LinearRegression

regressorobj = LinearRegression()
regressorobj.fit(X_train,y_train)
    
print(dataset)
# Predicting the Test set results
y_pred = regressorobj.predict(X_test)

# Visualise the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressorobj.predict(X_train), color = 'blue')
plt.title('Salary vs Years of Exp (Training data)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()


# Visualise the testing set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressorobj.predict(X_train), color = 'blue')
plt.title('Salary vs Years of Exp (Testdata)')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
plt.show()














