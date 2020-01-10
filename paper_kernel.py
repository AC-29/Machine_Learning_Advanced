import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import sys
class extension_cluster_kernel:
    def __init__(self, data, type, parameters=None):
        self.dict = {}
        self.data = data
        for i,d in enumerate(data):
            self.dict[str(d)] = i
        K = rbf_kernel(data)
        self.rbfK = K
        self.invRbfK = np.linalg.inv(self.rbfK)
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
    def poly_step(self,parameter_list):
        r = parameter_list[0]
        p = parameter_list[1]
        q = parameter_list[2]
        tmp_eig = np.zeros((len(self.eigvalues),len(self.eigvalues)))
        for i,e in enumerate(self.eigvalues):
            if i >= r:
                tmp_eig[i,i] = np.power(e,p)
            else:
                tmp_eig[i,i] = np.power(e,q)
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K
    def distance(self,X1,X2):
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        index2list = np.zeros((X2.shape[0]),dtype=int)
        for j, x2 in enumerate(X2):
            index2list[j] = self.dict[str(x2)]
        for i, x1 in enumerate(X1):
            ind1 = self.dict[str(x1)]
            gram_matrix[i, :] = self.K[ind1,index2list]
        return gram_matrix
    def distanceGeneral(self,X1,X2):
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        index2list = np.zeros((X2.shape[0]),dtype=int)
        for j, x2 in enumerate(X2):
            if str(x2) in self.dict:
                index2list[j] = self.dict[str(x2)]
            else:
                return self.distance_test(X1,X2)
        for i, x1 in enumerate(X1):
            if str(x1) in self.dict:
                ind1 = self.dict[str(x1)]
                gram_matrix[i, :] = self.K[ind1,index2list]
            else:
                return self.distance_test(X2,X1).T
        return gram_matrix
    def distance_test(self,X1,X2):
        index1list = np.zeros((X1.shape[0]),dtype=int)
        for j, x1 in enumerate(X1):
            index1list[j] = self.dict[str(x1)]
        known_samples = self.data[index1list]
        new_samples = X2
        V = rbf_kernel(new_samples,known_samples)
        K_hat_cut = np.zeros((len(index1list),len(index1list)))
        invRbfK_cut = np.zeros((len(index1list),len(index1list)))
        for i,ei in enumerate(index1list):
            for j,ej in enumerate(index1list):
                K_hat_cut[i,j] = self.K[ei,ej]
                invRbfK_cut[i,j] = self.invRbfK[ei,ej]
        projection = ((K_hat_cut).dot(invRbfK_cut)).dot(V.T)
        return projection
        
        
class random_walk:
    def __init__(self,labeledData,labels,unlabeledData,gam=None,k=1,t=2):
        data = np.concatenate((labeledData,unlabeledData))
        L = labeledData.shape[0]
        N = data.shape[0]
        labels = (1==labels)*1
        self.W = rbf_kernel(data,gamma=gam)
        for i in range((self.W).shape[0]):
            sort = list(np.argsort(self.W[i]))
            sort.remove(i)
            for j in sort[k:]:
                self.W[i,j]=0
                
        self.W = np.maximum(self.W,(self.W).transpose())
        self.P=np.zeros(((self.W).shape))
        sumRowsW = np.sum(self.W,axis=1)
        for i in range((self.W).shape[0]):
            self.P[i]=self.W[i]/sumRowsW[i]
        
        self.PT = np.linalg.matrix_power(self.P,t)
        
        self.labelProbability = np.zeros((N,2))
        #Initializing random values
        for i in range(N):
            number = (0.5 - np.random.rand())*0.4
            self.labelProbability[i,0] = 0.5+number
            self.labelProbability[i,1]=1-self.labelProbability[i,0]
        
        self.probability = np.zeros((N,L))
        oldloglike = -np.inf
        self.loglike = np.zeros(100)
        for iter in range(100):
            #E Step
            for i in range(N):
                for j in range(L):
                    self.probability[i,j] = self.labelProbability[i,labels[j]]*self.PT[i,j]
                    
            #M Step
            for i in range(N):
                self.labelProbability[i,0] = (np.sum((labels==0)*self.probability[i]))/(np.sum(self.probability[i]))
                self.labelProbability[i,1] = (np.sum((labels==1)*self.probability[i]))/(np.sum(self.probability[i]))
            self.loglike[iter] = np.sum(np.log(np.sum(self.probability,axis=0)))
            if np.abs(self.loglike[iter] - oldloglike) < 10**(-4):
                break
            oldloglike = self.loglike[iter]
        
        self.posterior = np.matmul((self.labelProbability).transpose(),self.PT)
        self.results = ((np.argmax(self.posterior,axis = 0))*2)-1
        
def JH_bound(dataTrain,targets,dataTest):
    data = np.concatenate((dataTrain,dataTest))
    kern = extension_cluster_kernel(data,'linear')
    linearEigval = -np.sort(-kern.eigvalues)[:(dataTrain.shape[0])]
    T = np.zeros((len(linearEigval)))
    for i,lamda_cut in enumerate(linearEigval):
        kern.poly_step([lamda_cut,1/2,2])
        clf = svm.SVC(C=100,kernel=kern.distance)
        clf.fit(dataTrain,targets)
        alphas = np.abs(clf.dual_coef_).reshape(-1)
        supportV = clf.support_
        for j in range(len(alphas)):
            T[i] += ((alphas[j]*kern.K[supportV[j],supportV[j]]-1)>0)*1
        T[i] = T[i]/dataTrain.shape[0]
    return T
        
