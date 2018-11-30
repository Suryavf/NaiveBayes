#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:31:46 2018

@author: victor
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.naive_bayes     import GaussianNB
from scipy.stats import norm
import seaborn as sns

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")

"""
Importing the dataset
---------------------
"""
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


"""
Splitting the dataset into the Training set and Test set
--------------------------------------------------------
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


"""
Feature Scaling
---------------
"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


ageNo     = [a[0] for b,a in zip(y_train,X_train) if b==0]
ageYes    = [a[0] for b,a in zip(y_train,X_train) if b==1]
SalaryNo  = [a[1] for b,a in zip(y_train,X_train) if b==0]
SalaryYes = [a[1] for b,a in zip(y_train,X_train) if b==1]

bins = 50; xlim = [-4,4]; ylim = [0,1.4]

plt.figure(figsize=(8,8))
plt.subplot(221)
sns.distplot(ageNo,bins=bins)
plt.xlim(xlim); plt.ylim(ylim)
plt.ylabel('No')

plt.subplot(222)
sns.distplot(SalaryNo,bins=bins)
plt.xlim(xlim); plt.ylim(ylim)

plt.subplot(223)
sns.distplot(ageYes,bins=bins)
plt.xlim(xlim); plt.ylim(ylim)
plt.ylabel('Yes'); plt.xlabel('Age')

plt.subplot(224)
sns.distplot(SalaryYes,bins=bins)
plt.xlim(xlim); plt.ylim(ylim)
plt.xlabel('Salary')


class gaussianFunction:
    def __init__(self,mean,std):
        self.mean = mean
        self. std = std
    
    def __getitem__(self,x):
        return (1/(self.std*np.sqrt(2*np.pi))) * np.exp( -(x-self.mean)*(x-self.mean)/( 2*self.std*self.std ) )
    
    
def naiveBayes(func,prob,X):
    n_class = len(prob)
    y_pred  = list()
    
    for x in X:
        scores = list()
        
        for c in range(n_class):    
            likelihood = [ gauss[feature]  for feature,gauss in zip(x,func[c]) ]
            likelihood = np.prod(likelihood)
            prior = prob[c] 
        
            scores.append(prior*likelihood)
        
        y_pred.append( np.argmax(scores) )
    
    return y_pred



mean,std = norm.fit(   ageNo ); funAN = gaussianFunction(mean,std)
mean,std = norm.fit(   ageYes); funAY = gaussianFunction(mean,std)
mean,std = norm.fit(SalaryNo ); funSN = gaussianFunction(mean,std)
mean,std = norm.fit(SalaryYes); funSY = gaussianFunction(mean,std)

probY = np.sum(y_train)/len(y_train)
probN = 1 - probY

func = [ [funAN,funSN],[funAY,funSY] ]
prob = [     probN    ,    probY     ]


y_pred = naiveBayes(func,prob,X_test)
acc = [int(a==b)  for a,b in zip(y_test,y_pred)]
print( 'Accuracy:', np.sum(acc)*100/len(acc),'%' )


def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3)
        for i in range(len(table[0]))
    ]
    row_format = "".join(["{:>" + str(longest_col) + "}" for longest_col in longest_cols])
    for row in table:
        print(row_format.format(*row))

TP = np.sum([int(a==0 and b==0)  for a,b in zip(y_test,y_pred)])
FP = np.sum([int(a==1 and b==0)  for a,b in zip(y_test,y_pred)])
FN = np.sum([int(a==0 and b==1)  for a,b in zip(y_test,y_pred)])
TN = np.sum([int(a==1 and b==1)  for a,b in zip(y_test,y_pred)])

table = [ [" ", "P", "N"],
          ["T",  TP,  FP],
          ["F",  FN,  TN] ]

