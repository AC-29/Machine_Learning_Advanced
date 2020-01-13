import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
import sys

class kernel_function:
    def __init__(self,kernelname, **parameter_list):
        self.kernel = kernelname
        self.params = parameter_list
    def calculate(self,data1,data2=None):
        if self.kernel == 'rbf':
            if self.params:
                return rbf_kernel(data1,data2,**self.params)
            else:
                return rbf_kernel(data1,data2)
        elif self.kernel == 'polynomial':
            if self.params:
                return polynomial_kernel(data1,data2,**self.params)
            else:
                return polynomial_kernel(data1,data2)
        elif self.kernel == 'knn':
            return knn_kernel(data1,**self.params)
        elif self.kernel == 'spline':
            return spline_kernel(data1,data2)

class kernel_mix:
    def __init__(self, *kernels):
        self.kernel_array = []
        for kernel in kernels:
            self.kernel_array.append(kernel)
    def calculate(self,data1,data2=None):
        tmp = self.kernel_array[0].calculate(data1,data2)
        ans = tmp/np.mean(tmp)
        for remaining_kernels in self.kernel_array[1:]:
            tmp = remaining_kernels.calculate(data1,data2)
            ans+= tmp/np.mean(tmp)
        return ans

def knn_kernel(data,k=2,t=4): #only without projection
    W = rbf_kernel(data)
    for i in range((W).shape[0]):
        sort = list(np.argsort(W[i]))
        sort.remove(i)
        for j in sort[k:]:
            W[i,j]=0
            
    W = np.maximum(W,(W).transpose())
    P=np.zeros(((W).shape))
    sumRowsW = np.sum(W,axis=1)
    for i in range((W).shape[0]):
        P[i]=W[i]/sumRowsW[i]
    PT = np.linalg.matrix_power(P,t)
    return 1/(PT + sys.float_info.epsilon)

def spline_kernel(data1,data2=None):
    if data2 is None:
        data2 = data1
    output = np.zeros((data1.shape[0],data2.shape[0]))
    for i,sample1 in enumerate(data1):
        for j,sample2 in enumerate(data2):
            tmp = np.minimum(sample1,sample2)
            output[i,j] = np.prod(1 + sample1*sample2 + sample1*sample2*tmp - (sample1+sample2)*(tmp**2)/2 + (tmp**3)/3)
    return output

class extension_cluster_kernel:
    def __init__(self, data, kernel_f, type, parameters=None):
        self.dict = {}
        self.data = data
        for i,d in enumerate(data):
            self.dict[d.tobytes()] = i
        self.kernel_function = kernel_f
        K = self.kernel_function.calculate(data)
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
    def linear_step(self,r):
        tmp_eig = np.diag(self.eigvalues)
        for i,e in enumerate(self.eigvalues):
            if e < r:
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
        cut_off = parameter_list[0]
        p = parameter_list[1]
        q = parameter_list[2]
        tmp_eig = np.zeros((len(self.eigvalues),len(self.eigvalues)))
        for i,e in enumerate(self.eigvalues):
            if e>=cut_off:
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
            index2list[j] = self.dict[x2.tobytes()]
        for i, x1 in enumerate(X1):
            ind1 = self.dict[x1.tobytes()]
            gram_matrix[i, :] = self.K[ind1,index2list]
        return gram_matrix
    def distanceGeneral(self,X1,X2):
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        index2list = np.zeros((X2.shape[0]),dtype=int)
        for j, x2 in enumerate(X2):
            if x2.tobytes() in self.dict:
                index2list[j] = self.dict[x2.tobytes()]
            else:
                return self.distance_test(X1,X2)
        for i, x1 in enumerate(X1):
            if x1.tobytes() in self.dict:
                ind1 = self.dict[x1.tobytes()]
                gram_matrix[i, :] = self.K[ind1,index2list]
            else:
                #print('I am in second branch')
                return self.distance_test(X2,X1).T
        return gram_matrix
    def distance_test(self,X1,X2):
        index1list = np.zeros((X1.shape[0]),dtype=int)
        for j, x1 in enumerate(X1):
            index1list[j] = self.dict[x1.tobytes()]
        new_samples = X2
        #print('multiplication of kernels', np.dot(self.K,self.invRbfK))
        V = self.kernel_function.calculate(new_samples,self.data)
        temp = ((self.K).dot(self.invRbfK)).dot(V.T)
        #print(temp.shape)
        projection = np.zeros((X1.shape[0],X2.shape[0]))
        for i in range(X1.shape[0]):
            ind1 = self.dict[X1[i].tobytes()]
            projection[i] = temp[ind1]
        return projection
        
        
class random_walk:
    def __init__(self,labeledData,labelss,unlabeledData,gam=None,k=1,t=2):
        data = np.concatenate((labeledData,unlabeledData))
        L = labeledData.shape[0]
        N = data.shape[0]
        labels = (1==labelss)*1
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
            if i < L:
                self.labelProbability[i,0] = (labels[i]==0)*1
                self.labelProbability[i,1] = (labels[i]==1)*1
            else:
                self.labelProbability[i,0] = 0.5
                self.labelProbability[i,1] = 0.5
            #number = (0.5 - np.random.rand())*0.4
            #self.labelProbability[i,0] = 0.5+number
            #self.labelProbability[i,1]=1-self.labelProbability[i,0]
        
        self.probability = np.zeros((N,L))
        oldloglike = -np.inf
        self.loglike = np.zeros(100)
        for iter in range(50):
            #E Step
            self.loglike[iter] = 0
            for j in range(L):
                tmpsum=0
                for i in range(N):
                    self.probability[i,j] = self.labelProbability[i,labels[j]]*self.PT[i,j] +sys.float_info.epsilon
                    tmpsum+=self.probability[i,j]
                self.probability[:,j] = self.probability[:,j]/(tmpsum + N*sys.float_info.epsilon)
                self.loglike[iter]+=np.log(tmpsum)
            #M Step
            for i in range(N):
                if np.sum(self.probability[i]) == 0:
                    self.labelProbability[i,0] = 0.5
                    self.labelProbability[i,1] = 0.5
                else:
                    self.labelProbability[i,0] = (np.sum((labels==0)*self.probability[i]))/(np.sum(self.probability[i]))
                    self.labelProbability[i,1] = (np.sum((labels==1)*self.probability[i]))/(np.sum(self.probability[i]))
            self.loglike[iter] = np.sum(np.log(np.sum(self.probability,axis=0)))
            if np.abs(self.loglike[iter] - oldloglike) < 10**(-4):
                print("RW converged, iter: " + str(iter))
                break
        self.posterior = np.zeros((2,N))
        #self.posterior = np.matmul((self.labelProbability).transpose(),self.PT)
        for k in range(N):
            for i in range(N):
                self.posterior[0,k]+=self.labelProbability[i,0]*self.PT[k,i]
                self.posterior[1,k]+=self.labelProbability[i,1]*self.PT[k,i]
                
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
        