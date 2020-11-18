# -*- coding: utf-8 -*-
"""
Justin Clark
CSYS 300
Final Project
popularityPrediction.py

Use different ML methods to predict song popularity 
Outline:
    
"""

### 1. Imports ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from scipy.stats import randint as sp_randint

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor 
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("merged_df.csv")
rows = data.shape[0]
cols = data.shape[1]
target_index = data.columns.get_loc("popularity")
X = data.iloc[:,target_index + 1:cols-2]
Y = data.iloc[:,target_index]
X = np.matrix(X)
Y = np.matrix(Y).T

# Distribution of Target Values
avg_pop = np.mean(Y)
std = np.std(Y)
plt.hist(Y,bins = 50)
plt.text(20,31,"Mean: {:.2f} Std: {:.2f}".format(avg_pop,std),fontsize = 14)
plt.grid(axis = 'y',alpha = 0.75)
plt.xlabel("Target Value: Song Popularity Score",fontsize = 18)
plt.ylabel("Frequency",fontsize = 18)
plt.title("Distribution of Target Values: Song Popularity Scores",fontsize = 18)
plt.show()


#X = preprocessing.standardize(X)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2)
X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.2)


clf = linear_model.Lasso(alpha = .01)
clf.fit(X_train,y_train)
lasso_pred = clf.predict(X_test)
MSE = mean_squared_error(y_test,lasso_pred)
print("Linear Regression + Lasso Regularization MSE: {}".format(MSE))
print("Lasso R2: {}".format(clf.score(X,Y)))
print(clf.coef_)
print(len(clf.coef_))