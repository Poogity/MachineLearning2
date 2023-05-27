import scipy.io
import numpy as np
import matplotlib.pyplot as plt

inputSize = 10

def ReLU(W1):
    Z1 = np.empty((128,1))
    for i in range(128):      
        Z1[i] = max(W1[i],0)
    return Z1

def Sigmoid(W2):
  X = 1/(1 + np.exp(W2))
  return X


#Loading Data
mat = scipy.io.loadmat('data21.mat')
A1 = mat["A_1"]
A2 = mat["A_2"]
B1 = mat["B_1"]
B2 = mat["B_2"]

for k in range(100):
    #Input is 10 elements of independant Gaussian random variables
    Z = np.empty((inputSize,1))
    for i in range(inputSize):
        Z[i] = np.random.normal(0,1)
    
    #forward propegation
    W1 = np.dot(A1,Z) + B1
    Z1 = ReLU(W1)
    W2 = np.dot(A2,Z1) + B2
    X = Sigmoid(W2)
    X2D = np.reshape(X, (28,28),order='F')
    plt.subplot(10,10,k+1),plt.imshow(X2D, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
plt.show()