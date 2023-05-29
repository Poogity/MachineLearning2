import scipy.io
import matplotlib.pyplot as plt
import numpy as np

inputSize = 10
n_iters = 1000
c= 10**(-5)
N=49

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
    lr = 0.05
    X,W1,W2 = forwProp(Z)    
    TX = np.dot(T,X)
    χη = np.reshape(TX,(7,7),order='F')
    #Xn = np.reshape(Xn,(7,7),order='F')
    #plt.show()
    U2 = np.dot(2*T.T/np.linalg.norm((TX-Xn)**2),TX-Xn)
    dSigmoid = -(np.exp(W2))/((1+np.exp(W2)))**2
    V2= (U2*dSigmoid)
    U1 = np.dot(A2.T,V2)
    V1 = U1 *(W1 > 0)*1 
    U0 = np.dot(A1.T,V1)
    gradJ = N*U0+2*Z
    J = (N*np.log(np.linalg.norm(TX-Xn)**2)+np.linalg.norm((Z)**2))
    #print(f'cost value = {J:.4f}')
    P = (1-lamda)*P + lamda*(gradJ**2)
    
    Z = Z - lr*gradJ/np.sqrt(c+P)
    return Z,P,J




#Loading Data
mat = scipy.io.loadmat('data21.mat')
A1 = mat["A_1"]
A2 = mat["A_2"]
B1 = mat["B_1"]
B2 = mat["B_2"]


#FORWARD PROPEGATION
#Input is 10 elements of independant Gaussian random variables
Z = np.empty((inputSize,1))


mat = scipy.io.loadmat('data23.mat')
XN = mat["X_n"]
XI = mat["X_i"]

T = np.zeros((49,784))
for i in range(49):
    for j in range(4):
        for k in range(4):
            T[i][28*j+k+i*4+ 3*28*(i//7)] = 1/16
            
cAA = np.empty((4,n_iters))
for k in range(4):
    minJ = float("inf")
    Xn = XN[:,k]
    Xn=np.reshape(Xn,(49,1))
    
    #initialize Z
    Z = np.empty((inputSize,1))
    for i in range(inputSize):
        Z[i] = np.random.normal(0,1)
    
    P= np.zeros((10,1))
    
    costArray = np.empty(n_iters)
    #5 different initializations, choosing 
    #the one with the smallest mean cost
    for j in range(5): 
        #initialize Z
        Z = np.empty((inputSize,1))
        for i in range(inputSize):
            Z[i] = np.random.normal(0,1)
    
        P= np.zeros((10,1))
        
        tempCostArray = np.empty(n_iters)
        for i in range(n_iters):
            Z,P,J=gradDescent(Z,P)
            tempCostArray[i] = J
        if (np.mean(tempCostArray[-20:])-minJ<0): 
            minJ = float(np.mean(tempCostArray[-20:]))
            bestZ=Z
            costArray = tempCostArray
        
    R,_,_ = forwProp(bestZ)
    R = np.reshape(R, (28,28), order='F')
    cAA[k] = costArray
    Xn=np.reshape(Xn,(7,7),order='F')
    Xn=np.kron(Xn,np.ones((4,4)))
    Xi=np.reshape(XI[:,k],(28,28),order='F')

    plt.subplot(4,3,3*k+1),plt.imshow(Xi, cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,3*k+2),plt.imshow(Xn, cmap = 'gray')
    plt.title('Noisy'), plt.xticks([]), plt.yticks([])
    plt.subplot(4,3,3*k+3),plt.imshow(R, cmap = 'gray')
    plt.title('Restored'), plt.xticks([]), plt.yticks([])
plt.show()

xaxis = np.linspace(1, n_iters,n_iters)
for i in range(4):  
    plt.plot(xaxis,cAA[i])   