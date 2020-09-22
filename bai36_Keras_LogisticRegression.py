# -*- coding: utf-8 -*-
"""
Created on Thu May 14 23:22:03 2020

@author: phamk
"""


import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import losses
from keras import optimizers

# 1. Prepare data 
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# 2. Build model 
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation('sigmoid'))

# 3. gradient descent optimizer and loss function 
sgd = optimizers.SGD(lr=0.05)
model.compile(loss=losses.binary_crossentropy, optimizer=sgd)

# 4. Train the model 
model.fit(X, y, epochs=3000, batch_size=1) 

print(model.get_weights())