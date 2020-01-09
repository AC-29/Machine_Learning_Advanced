import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import random
from tsvm import *
from sklearn import metrics
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import sys

def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size


def is_float(string):
    """ True if given string is float else False"""
    try:
        return float(string)
    except ValueError:
        return False


def load_file(file):
    data = []
    with open(file, 'r') as f:
        d = f.readlines()
        for i in d:
            k = i.rstrip().split(",")
            data.append([float(i) if is_float(i) else i for i in k])

    data = np.array(data, dtype='O')
    return data


def processing(data):
    for i in range(len(data)): # -1 is assigned to 'negative', +1 is assigned to 'positive'
        if (data[i][10] == ' negative'):
            data[i][10] = -1
        else:
            data[i][10] = 1

    i = 0
    train_data = []
    train_target = []
    for i in range(len(data)):
        train_data.append(data[i][0:(len(data[0]) - 1)])
        train_target.append(data[i][10])
    return train_data, train_target





class extension_cluster_kernel:
    def __init__(self, data, type, parameters=None):
        self.dict = {}
        for i, d in enumerate(data):
            self.dict[str(d)] = i
        sigma = 6
        gam = 1 / (2 * sigma ** 2)
        K = rbf_kernel(data, gamma=gam)
        D = np.zeros((K.shape[0], K.shape[1]))
        diagonalElements = np.sum(K, axis=1)
        for i in range(D.shape[0]):
            D[i, i] = diagonalElements[i] ** (-0.5)
        L = (D.dot(K)).dot(D)
        self.L = L
        self.eigvalues, self.eigvectors = np.linalg.eigh(L)
        if parameters:
            self.K = getattr(self, type)(parameters)
        else:
            self.K = getattr(self, type)()

    def compute_K(self, L):
        D_hat = np.diag((1. / (np.diagonal(L) + sys.float_info.epsilon)) ** (0.5))
        K_hat = (D_hat.dot(L)).dot(D_hat)
        return K_hat

    def linear(self):
        return self.compute_K(self.L)

    def step(self, lambda_cut):
        tmp_eig = np.zeros((len(self.eigvalues), len(self.eigvalues)))
        for i, e in enumerate(self.eigvalues):
            if e >= lambda_cut:
                tmp_eig[i, i] = 1
            else:
                tmp_eig[i, i] = 0
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K

    def linear_step(self, r):
        tmp_eig = np.diag(self.eigvalues)
        for i, e in enumerate(self.eigvalues):
            if e < r:
                tmp_eig[i, i] = 0
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K

    def polynomial(self, t):
        tmp_eig = np.diag(np.power(self.eigvalues, t))
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K

    def poly_step(self, parameter_list):
        cut_off = parameter_list[0]
        p = parameter_list[1]
        q = parameter_list[2]
        tmp_eig = np.zeros((len(self.eigvalues), len(self.eigvalues)))
        for i, e in enumerate(self.eigvalues):
            if e >= cut_off:
                tmp_eig[i, i] = np.power(e, p)
            else:
                tmp_eig[i, i] = np.power(e, q)
        L_hat = (self.eigvectors.dot(tmp_eig)).dot(self.eigvectors.T)
        self.K = self.compute_K(L_hat)
        return self.K

    def distance(self, X1, X2):
        gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
        index2list = np.zeros((X2.shape[0]), dtype=int)
        for j, x2 in enumerate(X2):
            index2list[j] = self.dict[str(x2)]
        for i, x1 in enumerate(X1):
            ind1 = self.dict[str(x1)]
            gram_matrix[i, :] = self.K[ind1, index2list]
        return gram_matrix

def k_th_largest_eig(eig,k):
    arr = eig.copy()
    arr.sort()
    return arr[-k]


# load training dataset
train_dataset = load_file('page-blocks0-5-1tra.dat')     # https://sci2s.ugr.es/keel/imbalanced.php#sub10
train_data,train_target = processing(train_dataset)

#load test_dataset
test_dataset = load_file('page-blocks0-5-1tst.dat') # https://sci2s.ugr.es/keel/imbalanced.php#sub10
test_data, test_target = processing(test_dataset)

# Train SVM and compute the accuracy
clf = svm.SVC(C = 0.000001, gamma=0.1)
clf.fit(train_data, train_target)
score = clf.score(test_data, test_target)
print("SVM score = ", score)




###### TSVM
train_dataset_label=[]
train_dataset_unlabeled=[]
train_dataset = load_file('page-blocks0-5-1tra.dat')
i=0
for i in range(len(train_dataset)): # 20% of the labeled dataset is used for training
    if (random.uniform(0, 1) <= 0.2):
        train_dataset_label.append(train_dataset[i])
    else:
        train_dataset_unlabeled.append(train_dataset[i])

train_data_TSVM, train_target_TSVM = processing((train_dataset_label))
train_data_unlabeled_TSVM, train_target_unlabeled_TSVM= processing(train_dataset_unlabeled)

########## Run TSVM
#np.random.seed(8)
#Hyperparameters
C = 100
gamma = 0.1
kernel = 'linear'
model = TSVM()
model.initial(kernel, gamma, C)
model.train(train_data_TSVM[0:150], train_target_TSVM[0:150], train_data_unlabeled_TSVM[0:1000])
#print (train_target_TSVM[0:150])
# Test TSVM
test_predictions = model.predict(test_data)

# performance
accuracy = compute_accuracy(test_predictions, test_target)
f1_score = metrics.f1_score(test_target, test_predictions, average='macro')
print("TSVM Accuracy = ", accuracy)
#print("F1 score = ", f1_score)



### cluster kernel
trn_labeled = np.array(train_data_TSVM[0:150])
trn_unlabeled = np.array(train_data_unlabeled_TSVM[0:1000])
trn = np.concatenate((trn_labeled, trn_unlabeled),axis = 0)

tes = np.array(test_data)

data = np.concatenate((trn, tes),axis = 0)
#np.savetxt('trn.txt', trn, delimiter=',')
#np.savetxt('tes.txt', tes, delimiter=',')

lin_ker = extension_cluster_kernel(data,'linear')
#eigenvalue
eig = lin_ker.eigvalues


cut_off = k_th_largest_eig(eig,10)
lin_ker.poly_step([cut_off,1/2,2])
clf = svm.SVC(kernel = lin_ker.distance, C=100, class_weight='balanced')
print (lin_ker.K)




trn_target = np.array(train_target_TSVM[0:150])
#np.savetxt('trn_target.txt', trn_target, delimiter=',')
#np.savetxt('trn_labeled.txt', trn_labeled, delimiter=',')

#clf.fit(trn_labeled, trn_target)    #it does not work
#print (clf.score(tes, test_target ))


