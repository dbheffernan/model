# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 20:29:42 2018

@author: Dillon
"""
import numpy as np
from sklearn import preprocessing
from sklearn import metrics

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

bigexpend.Expense_Purpose = bigexpend.Expense_Purpose.fillna('')
bigexpend.Payee_Name = bigexpend.Payee_Name.fillna('')
bigexpend.Payee_Type = bigexpend.Payee_Type.fillna('')
bigexpend.Expense_Category = bigexpend.Expense_Category.fillna('')
bigexpend.Amount = bigexpend.Amount.fillna(0)
bigexpend.Expenditure_Date = pd.to_datetime(bigexpend.Expenditure_Date)
bigexpend['Month'] = bigexpend['Expenditure_Date'].dt.month
bigspend = bigexpend
le = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le3 = preprocessing.LabelEncoder()

bigspend.Expense_Category = bigspend.Expense_Category.fillna('')
le.fit(bigspend.Expense_Purpose)
le2.fit(bigspend.Payee_Type)
le3.fit(bigspend.Expense_Category)


bigspend['num'] = le3.transform(bigspend.Expense_Category)
bigspend['type'] = le2.transform(bigspend.Payee_Type)
bigspend = bigspend[bigspend.Expense_Category != '']
bx = bigspend.loc[bigspend.Expense_Category != 'X']


 
features= [c for c in bx.columns.values if c in ['Payee_Name','type','Amount','Month']]
target = 'num'

X_train, X_test, y_train, y_test = train_test_split(bx[features],bx[target])

payee = Pipeline([
                ('selector', TextSelector(key='Payee_Name')),
                ('vect', CountVectorizer(ngram_range=(1, 3), stop_words='english'))
            ])
purpose = Pipeline([
                ('selector', TextSelector(key='Expense_Purpose')),
                ('vect', CountVectorizer( stop_words='english'))
            ])
    
amount = Pipeline([
                ('selector', NumberSelector(key='Amount')),
                ('standard', StandardScaler())
            ])

feats = FeatureUnion([
        ('payee',payee),
        #('purpose',purpose),
        ('type',NumberSelector(key='type')),
        ('amount',amount),
        ('month',NumberSelector(key='Month'))])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)

pipeline = Pipeline([
    ('features',feats),
    ('classifier', LogisticRegression()),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)
print(np.mean(preds == y_test))

pipeline2 = Pipeline([
    ('features',feats),
    ('classifier', LinearSVC()),
])
    
pipeline2.fit(X_train, y_train)
preds = pipeline2.predict(X_test)
print(np.mean(preds == y_test))
