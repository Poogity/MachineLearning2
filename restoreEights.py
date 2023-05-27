import scipy.io
import matplotlib.pyplot as plt
import numpy as np

inputSize = 10
N=500

def ReLU(W1):
    Z1 = np.empty((128,1))
    for i in range(128):      
        Z1[i] = max(W1[i],0)
    return Z1

def Sigmoid(W2):
  X = 1/(1 + np.exp(W2))
  return X
def forwProp(Z):   
    W1 = np.dot(A1,Z) + B1
    Z1 = ReLU(W1)
    W2 = np.dot(A2,Z1) + B2
    X = Sigmoid(W2)
    return X,W1,W2

def gradDescent(Z,P):
    lamda = 0.01
    if P.all() ==0: lamda=1
    lr = 0.01
    Zimg,W1,W2 = forwProp(Z)
    U2 = np.dot(2*T,1/((TX-Xn)+c))
    dSigmoid = ((-np.exp(W2))/(1+np.exp(W2)))**2
    V2= (U2*dSigmoid)
    U1 = np.dot(A2.T,V2)
    V1 = U1 *(W1 > 0)*1  
    U0 = np.dot(A1.T,V1)
    gradJ = N*U0+2*Z
    J = np.mean(N*np.log(abs(TX-Xn+c)**2)+abs(Zimg)**2)
    P = (1-lamda)*P + lamda*(gradJ**2)
    
    Z = Z - lr*gradJ/(c+P)
    return Z,P,Zimg


c= 10**(-5)


#Loading Data
mat = scipy.io.loadmat('data21.mat')
A1 = mat["A_1"]
A2 = mat["A_2"]
B1 = mat["B_1"]
B2 = mat["B_2"]


#FORWARD PROPEGATION
#Input is 10 elements of independant Gaussian random variables
Z = np.empty((inputSize,1))
for i in range(inputSize):
    Z[i] = np.random.normal(0,1)
X,_,_ = forwProp(Z) #Generative Model

mat = scipy.io.loadmat('data22.mat')
XN = mat["X_n"]
XI = mat["X_i"]


T = np.eye(N,784,order='C')
T=np.vstack((T,np.zeros((784-N,784))))

for k in range(4):
    Xn = np.dot(T,XN[:,k])
    Xn=np.reshape(Xn,(784,1))
    TX = np.dot(T,X)
    
    #initialize Z
    Z = np.empty((inputSize,1))
    for i in range(inputSize):
        Z[i] = np.random.normal(0,1)
    
    P= np.zeros((10,1))    
    for i in range(20):
        Z,P,Zimg=gradDescent(Z,P)
    
    
    #Z,_,_ = forwProp(Z)
    Z = np.reshape(Zimg, (28,28), order='F')
    
    
    
    Xn=np.reshape(Xn,(28,28),order='F' )
    Xi=np.reshape(XI[:,k],(28,28),order='F')
    plt.subplot(4,3,3*k+1),plt.imshow(Xi, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,3*k+2),plt.imshow(Xn, cmap = 'gray')
    plt.title('Noisy'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,3*k+3),plt.imshow(Z, cmap = 'gray')
    plt.title('Restored'), plt.xticks([]), plt.yticks([])

plt.show()
    #plt.imshow(TX,cmap='gray')
    #plt.xticks([]), plt.yticks([])