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