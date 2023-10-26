#!/usr/bin/env python
# coding: utf-8

# In[1]:


#k-Nearest Neighbors (k-NN)
#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


# In[2]:


#Step 1: Import data
file = "C:/Users/thoma/Dropbox/Queen Mary/Teaching/ECOM055 Risk Management for Banking/ECOM055_2022_23_SemC/Week 9/loans50k_v2.csv"
loans = pd.read_csv(file,  encoding = 'latin-1')
print(loans)


# In[3]:


#Split Data into Test and Training Data Sets
x_data = loans.drop('status', axis = 1) 
y_data = loans['status']# Use train_test_split function to generate training data and test  # data. Test data set is 30% of original data set. 
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.30, random_state=42)


# In[4]:


# Fit a KNeighborsClassifier to the data with k=1
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
# Predict the default for the new observations
y_pred = clf.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)
recall = recall_score(y_test, y_pred)
print("Recall Score:", recall)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:


# Create a Linear Regression model
regression_model = LinearRegression()

# Fit the model to the training data
regression_model.fit(X_train, y_train)

# Predict using the trained model
y_pred = regression_model.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
precision = precision_score(y_test, y_pred)
print("Precision Score:", precision)
recall = recall_score(y_test, y_pred)
print("Recall Score:", recall)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

