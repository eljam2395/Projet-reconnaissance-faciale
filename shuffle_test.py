import numpy as np
from sklearn.utils import shuffle


X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])


print (X)
print(y)

X, y = shuffle(X, y, random_state=0)

print(" \n")
print(X)
print(y)


print(X[1,:])
