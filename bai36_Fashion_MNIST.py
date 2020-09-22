# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:48:07 2020

@author: phamk
"""


# 1. prepare data 
import numpy as np
import keras 
#from __future__ import print_function 
from keras.datasets import fashion_mnist
from mnist.loader import MNIST

'''
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('x_train shape:\t', x_train.shape)
print('x_test shape:\t', x_test.shape)
print('y_train shape:\t', y_train.shape)
print('y_test shape:\t', y_test.shape)
'''

mndata = MNIST('E:/AI/example/fashion')
mndata.load_testing()
mndata.load_training()
x_test = mndata.test_images
x_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

x_test = np.asarray(x_test)
x_train = np.asarray(x_train)
y_test = np.asarray(y_test)
y_train = np.asarray(y_train)

x_test = x_test.reshape(10000, 28,28)
x_train = x_train.reshape(60000, 28,28)

print('x_train shape:\t', x_train.shape)
print('x_test shape:\t', x_test.shape)
print('y_train shape:\t', y_train.shape)
print('y_test shape:\t', y_test.shape)


# data normalization
x_train = x_train/255.
x_test = x_test/255. 
num_classes = 10 
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import metrics 
# 2. buid model 
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 3. loss, metrics 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              metrics=['accuracy'])
# 4. train the model 
model.fit(x_train, y_train, batch_size=128, epochs = 20)

from keras import metrics 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: %.4f'% score[0])
print('Test accuracy %.4f'% score[1])
