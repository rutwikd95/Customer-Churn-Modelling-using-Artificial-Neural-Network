# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:10:10 2018

@author: rutwi
"""

# PART 1 - Data Preprocessing

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Data Set
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Removing first column
X = X[:,1:]


# Splitting Dataset intro Training and Testing Dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#PART 2 - Now Lets make the ANN

#Importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers  import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layers
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 11))

#Adding the 2nd Hidden Layer
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu'))

#Adding the Output Layer
classifier.add(Dense(output_dim = 1, init = "uniform", activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

#PART 3 - Making the prediction and evaluating the model

#Predicting the test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 