# -*- coding: utf-8 -*-
"""
Created on Sat May 30 00:54:14 2020

@author: phamk
"""


import numpy as np
from numpy import linalg as LA

m, n = 2, 3
A = np.random.rand(m, n)

U, S, V = LA.svd(A)

# checking if U, V are orthogonal and S is a diagonal matrix with
# nonnegative decreasing elements
print ('Frobenius norm of (UU^T - I) =', \
    LA.norm(U.dot(U.T) - np.eye(m)))
print ('\n S = ', S, '\n')
print ('Frobenius norm of (VV^T - I) =', \
    LA.norm(V.dot(V.T) - np.eye(n)))