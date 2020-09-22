# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 01:56:23 2020

@author: phamk
"""


# %reset
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score

mntrain = MNIST('E:/AI/example/MNIST/')
mntrain.load_training()
Xtrain = np.asarray(mntrain.train_images)/255.0
ytrain = np.array(mntrain.train_labels.tolist())

mntest = MNIST('E:/AI/example/MNIST/')
mntest.load_testing()
Xtest = np.asarray(mntest.test_images)/255.0
ytest = np.array(mntest.test_labels.tolist())

# train
logreg = linear_model.LogisticRegression(C=1e5, 
        solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(Xtrain, ytrain)

# test
y_pred = logreg.predict(Xtest)
print ("Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist())))

Xtrain = np.concatenate((np.ones((Xtrain.shape[0], 1))/255.0, Xtrain), axis = 1)
Xtest = np.concatenate((np.ones((Xtest.shape[0], 1))/255.0, Xtest), axis = 1)

print(Xtrain.shape)
logreg = linear_model.LogisticRegression(C=1e5, solver = 'lbfgs', multi_class = 'multinomial')
logreg.fit(Xtrain, ytrain)

y_pred = logreg.predict(Xtest)
print ("Accuracy: %.2f %%" %(100*accuracy_score(ytest, y_pred.tolist())))