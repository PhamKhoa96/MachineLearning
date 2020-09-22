# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:14:50 2020

@author: phamk
"""


import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

A = sio.loadmat('myARgender.mat')
X_train = A['Y_train'].T 
X_test = A['Y_test'].T 
N = 700
y_train = A['label_train'].reshape(N)
y_test = A['label_test'].reshape(N)

clf = SVC(kernel='poly', degree = 3, gamma=1, C = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))