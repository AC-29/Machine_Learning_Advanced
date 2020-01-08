import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import sys
class extension_cluster_kernel:
    def __init__(self, data, type, parameters=None):
        self.dict = {}
        for i,d in enumerate(data):
            self.dict[str(d)] = i
        K = rbf_kernel(data)
        D = np.zeros((K.shape[0],K.shape[1]))
        diagonalElements = np.sum(K,axis = 1)
        for i in range(D.shape[0]):
            D[i,i] = diagonalElements[i]**(-0.5)
        L = (D.dot(K)).dot(D)
        self.L = L
        self.eigvalues,self.eigvectors = np.linalg.eigh(L)
        if parameters:
            self.K = getattr(self, type)(parameters)
        else:
            self.K = getattr(self, type)()
    def compute_K(self,L):
        D_hat = np.diag((1./(np.diagonal(L) + sys.float_info.epsilon))**(0.5))
        K_hat = (D_hat.dot(L)).dot(D_hat)
        return K_hat
    def linear(self):
        return self.compute_K(self.L)
    def step(self,lambda_cut):
        tmp_eig = np.zeros((len(self.eigvalues),len(self.eigvalues)))
        for i,e in enumerate(self.eigvalues):
            if e >= lambda_cut:
                tmp_eig[i,i] = 1
            else:
                tmp_eig[i,i] = 0
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K
    def linear_step(self,lambda_cut):
        tmp_eig = np.diag(self.eigvalues)
        for i,e in enumerate(self.eigvalues):
            if e < lambda_cut:
                tmp_eig[i,i] = 0
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K
    def polynomial(self,t):
        tmp_eig = np.diag(np.power(self.eigvalues,t))
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K
    def distance(self,x1,x2):
        ind1 = self.dict[str(x1)]
        ind2 = self.dict[str(x2)]
        return self.K[ind1,ind2]
    


    




def clusterKernel(dataTrain,dataUnlabeled,dataTest,k):
    
    #1 Compute affinity matrix, which is an RBF Kernel matrix but with
    #diagonal elements being 0 instead of 1.
    data = np.concatenate((dataTrain,dataUnlabeled,dataTest))
    affinityM = rbf_kernel(data)
    
    for i in range(affinityM.shape[0]):
        affinityM[i,i] = 0
                
    #2 Let D be a diagonal matrix with diagonal elemtns equal
    #to the sum of the rows (or the columns) of K and construct
    #the matrix L = D^(-1/2)*K*D^(-1/2). In this case, K (Kernel matrix)
    #will be affinityM
    D = np.zeros((affinityM.shape[0],affinityM.shape[1]))
    diagonalElements = np.sum(affinityM,axis = 1)
    
    for i in range(D.shape[0]):
        D[i,i] = diagonalElements[i]**(-0.5)
        
    aux = np.matmul(D,affinityM)
    L = np.matmul(aux,D)

    
    #3 Find the eigenvectors (v1,...,vk) of L corresponding the first k eigenvalues
    eValues,eVectors = np.linalg.eigh(L)
    eVectors = eVectors[:,0:k]
    
    #4 the new representation of the point xi is (vi1,....vik) and i snormalized
    #to have length one.
    newData = np.zeros((eVectors.shape[0],eVectors.shape[1]))
    for i in range(eVectors.shape[0]):
        denominator = np.sum([j**2 for j in eVectors[i,:]])
        newData[i,:] = [j/(denominator**(0.5)) for j in eVectors[i,:]]
        
    
    return newData