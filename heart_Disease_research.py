# -*- coding: utf-8 -*-
"""
Project for Dr Weissmans' patients in cardiology Yale University
Created on Wed Jun 13 13:21:05 2017

@author: chris morris
"""

import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import keras 
import urllib.request
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


print('python: {}'.format(sys.version))
print('pandas: {}'.format(pd.__version__))
print('Numpy: {}'.format(np.__version__))
print('sklearn: {}'.format(sklearn.__version__))

print('matplotlib: {}'.format(matplotlib.__version__))
print('keras: {}'.format(keras.__version__))



# this is the data I work with here. This is for training of the network. Other patients' data is very unique and protected.

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# the names will be the names of each column in our pandas DataFrame
names = ['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'class']

# read the csv file of information:
cleveland = pd.read_csv(url, names=names)


# I generally look at my data and play around with it so I know what it is: 
cleveland.shape
print('Shape of the dataframe: {}'.format(cleveland.shape)
#print(cleveland.loc[1])

# print the last 20 data points:

cleveland.loc[280:]


# I have some missing data, so I will pre-process it and fill in some blanks or get rid of them:

data= cleveland[~cleveland.isin(['?'])]
data.loc[280:]


# I need to dropt the rows with an 'NaN' values. This will throw the classification results off:

data = data.dropna(axis= 0)
data.loc[280:]

data.shape
data.dtypes

# some colums or attributes were kept as "objects". I have to turn them into numbers so I can furthur analyze the dataset

data= data.apply(pd.to_numeric)
print(data)
data.dtypes

# to know what I'm dealing with, I can also use functions
# that print out the characteritscs and some statistics of the data like them 'mean', 'standard deviation',
# the 'count' of all the data points in the columns, etc:

data.describe()


# Then many times, I have to present the data to people, so I can
# use matplot to better visualize the information so it's easier to understand:

data.hist(figsize = (12,12))
plt.show


# create X and y datasets for training

from sklearn import model_selection

# we drop the "class" attribute here, because we are goiing to use it for the target in the 'y'
X = np.array(data.drop(['class'],1))

# now this here is out "target" Or what we are trying to train the model with...
y= np.array(data['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)


# convert my data into categorical laberl for logistic regression classifcation:

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = None)
y_test = to_categorical(y_test, num_classes = None)

# I will show you the results here:

print(y_train.shape)
print(y_test[:10])


# now I'm going to build the neural network with several layers:
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# define a function to build the keras model:

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# define a function to build the keras model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    # compile model here using the "Adaptive Moment Estimation(Adam) optimizer" optimizer function 
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()

print(model.summary())

# now this is where I fit my model to the training data I have:


model.fit(X_train, y_train, epochs = 100, batch_size = 10)



# convert to BINARY classification problem, heart disease or NO heart disease:

y_train_binary = y_train.copy()
y_test_binary = y_test.copy()

y_train_binary[y_train_binary > 0]=1
y_test_binary[y_test_binary> 0] = 1


print(y_train_binary[:20])

# define a new keras model for binary classification
# define a function to build the keras model:

def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model here using the "Adaptive Moment Estimation(Adam) optimizer" optimizer function 
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())


# fit the binary model to my training data:

binary_model.fit(X_train, y_train_binary, epochs=100, batch_size=10, verbose = 1)

# This is my classification report using predictions for the categorical model:

from sklearn.metrics import classification_report, accuracy_score

categorical_prediction = np.argmax(model.predict(X_test), axis = 1)
categorical_prediction

# This will print out the results for "precision" or false positives. As well
#as "recall" or false negatives.
# also the "F1-score" or the combaintion of the precision and the recall scores

print('Results for Categorical Model')
print(accuracy_score(y_test, categorical_prediction))
print(classification_report(y_test, categorical_prediction))


# This is another report but for the BINARY model now:

binary_prediction = np.round(binary_model(X_test)).astype(int)


print("Results for my classification model: ")
print(accuracy_score(y_test_binary, binary_model))
print(classification_report(y_test_binary, binary_model))




































