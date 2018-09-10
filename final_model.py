# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 02:57:22 2018

@author: Dillon
"""


import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLarsIC

from sklearn.pipeline import make_pipeline


lasso_eps = 0.0001
lasso_nalpha=20
lasso_iter=50000

degree_min = 1
degree_max = 3


X = main.drop(['Percent'],axis=1)
Y = main.Percent
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=.8)

for degree in range(degree_min,degree_max+1):
    model = make_pipeline(
            PolynomialFeatures(degree, interaction_only=False), 
            LassoCV(eps=lasso_eps,
                    n_alphas=lasso_nalpha,
                    max_iter=lasso_iter,
                    normalize=True,
                    cv=20,n_jobs=-1,verbose=True,tol=0.001))
    
    
model.fit(X_train,y_train)
test_pred = np.array(model.predict(X_test))
RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
test_score = model.score(X_test,y_test)
    
    