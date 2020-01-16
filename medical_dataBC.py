import argparse
import os
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from tsvm import *
from sklearn.datasets import load_breast_cancer
from sklearn import datasets
from sklearn.semi_supervised import LabelSpreading
import sys
import timeit
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

gamma = 0.00001  # 0.00001 good


def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size


class extension_cluster_kernel:
    def __init__(self, data, type, parameters=None):
        self.dict = {}
        for i, d in enumerate(data):
            self.dict[str(d)] = i
        # sigma = 6
        # gam = 1 / (2 * sigma ** 2)
        gam = gamma
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


def k_th_largest_eig(eig, k):
    arr = eig.copy()
    arr.sort()
    return arr[-k]


class random_walk:
    def __init__(self, labeledData, labelss, unlabeledData, gam=None, k=1, t=2):
        data = np.concatenate((labeledData, unlabeledData))
        L = labeledData.shape[0]
        N = data.shape[0]
        labels = (1 == labelss) * 1
        self.W = rbf_kernel(data, gamma=gam)
        for i in range((self.W).shape[0]):
            sort = list(np.argsort(-self.W[i]))
            sort.remove(i)
            for j in sort[k:]:
                self.W[i, j] = 0

        self.W = np.maximum(self.W, (self.W).transpose())
        self.P = np.zeros(((self.W).shape))
        sumRowsW = np.sum(self.W, axis=1)
        for i in range((self.W).shape[0]):
            self.P[i] = self.W[i] / sumRowsW[i]

        self.PT = np.linalg.matrix_power(self.P, t)

        self.labelProbability = np.zeros((N, 2))
        # Initializing random values
        for i in range(N):
            if i < L:
                self.labelProbability[i, 0] = (labels[i] == 0) * 1
                self.labelProbability[i, 1] = (labels[i] == 1) * 1
            else:
                self.labelProbability[i, 0] = 0.5
                self.labelProbability[i, 1] = 0.5
            # number = (0.5 - np.random.rand())*0.4
            # self.labelProbability[i,0] = 0.5+number
            # self.labelProbability[i,1]=1-self.labelProbability[i,0]

        self.probability = np.zeros((N, L))
        oldloglike = -np.inf
        self.loglike = np.zeros(100)
        for iter in range(50):
            # E Step
            self.loglike[iter] = 0
            for j in range(L):
                tmpsum = 0
                for i in range(N):
                    self.probability[i, j] = self.labelProbability[i, labels[j]] * self.PT[
                        i, j] + sys.float_info.epsilon
                    tmpsum += self.probability[i, j]
                self.probability[:, j] = self.probability[:, j] / (tmpsum + N * sys.float_info.epsilon)
                self.loglike[iter] += np.log(tmpsum)
            # M Step
            for i in range(N):
                if np.sum(self.probability[i]) == 0:
                    self.labelProbability[i, 0] = 0.5
                    self.labelProbability[i, 1] = 0.5
                else:
                    self.labelProbability[i, 0] = (np.sum((labels == 0) * self.probability[i])) / (
                        np.sum(self.probability[i]))
                    self.labelProbability[i, 1] = (np.sum((labels == 1) * self.probability[i])) / (
                        np.sum(self.probability[i]))
            self.loglike[iter] = np.sum(np.log(np.sum(self.probability, axis=0)))
            if np.abs(self.loglike[iter] - oldloglike) < 10 ** (-4):
                print("RW converged, iter: " + str(iter))
                break
        self.posterior = np.zeros((2, N))
        # self.posterior = np.matmul((self.labelProbability).transpose(),self.PT)
        for k in range(N):
            for i in range(N):
                self.posterior[0, k] += self.labelProbability[i, 0] * self.PT[k, i]
                self.posterior[1, k] += self.labelProbability[i, 1] * self.PT[k, i]

        self.results = ((np.argmax(self.posterior, axis=0)) * 2) - 1

def processing(data):
    i = 0
    t_data = []
    t_target = []
    for i in range(len(data)):
        t_data.append(data[i][0:(len(data[0]) - 1)])
        t_target.append(data[i][30])
    return np.array(t_data,dtype=float), np.array(t_target,dtype=float)


dataset = load_breast_cancer()
dataset_features = np.array(dataset.data, dtype=float)
dataset_target = np.array(dataset.target, dtype=float)

for i in range(len(dataset_target)):
    if (dataset_target[i] == 0):
        dataset_target[i] = -1

print('number of -1:', dataset_target[dataset_target == -1].shape)
print('number of 1:', dataset_target[dataset_target == 1].shape)
print('total size', len(dataset_target))

# print (train_targets[train_targets==1].shape)
    # print (train_targets[train_targets==-1].shape)
# Define train, unlabeled and test dataset


C=10
iterations = 50
accuracy_total = np.zeros((iterations, 4))
kernel='rbf'
for i in range(iterations):
    np.random.seed(i)

    # indeces = np.random.permutation(dataset_features.shape[0])


    lengthTrain = 1

    while lengthTrain == 1:
        trainSize=40
        unlabeled_end = 140
        indeces = np.random.permutation(dataset_features.shape[0])
        train_targets = dataset_target[indeces[0:trainSize]]
        lengthTrain = len(np.unique(train_targets))

    # trainSize=10
    # unlabeled_end = 140
    train_l = dataset_features[indeces[0:trainSize]]
    train_targets = dataset_target[indeces[0:trainSize]]
    train_unlabeled = dataset_features[indeces[trainSize:unlabeled_end]]
    train_unlabeled_target = dataset_target[indeces[trainSize:unlabeled_end]]
    test_data = dataset_features[indeces[unlabeled_end:-1]]
    test_data_target = dataset_target[indeces[unlabeled_end:-1]]

    # test_with_tar = np.concatenate((test_data,test_data_target.reshape(-1,1)),axis=1)
    # test_with_tar = test_with_tar[test_with_tar[:, 30].argsort()]   # sort by label
    # minus_ones = test_with_tar[0:100]
    # ones = test_with_tar[300:400]
    #
    # even_test_data = np.concatenate((minus_ones,ones),axis=0)
    # test_data , test_data_target = processing(even_test_data)


    # for in range(len(test_data_target)):
    #     if (test_data_target[i]==1):
    #         ones.append(i)


    # print (train_targets[train_targets==1].shape)
    # print (train_targets[train_targets==-1].shape)





    ######################################################


    ######## SVM
    clf = svm.SVC(C=C, kernel=kernel, gamma=gamma)
    clf.fit(train_l, train_targets)

    accuracy_total[i, 0] = clf.score(test_data, test_data_target)
    # print("SVM score = ", score)

    #######################################################


    ##########  TSVM
    model = TSVM(kernel, gamma, C)

    train_unlabeled_plus_test_data = np.concatenate((train_unlabeled,test_data),axis=0)
    model.train(train_l, train_targets, train_unlabeled)


    # Test TSVM
    test_predictions = model.clf.predict(test_data)
    # performance
    accuracy_total[i, 1] = compute_accuracy(test_predictions, test_data_target)
    # tn, fp, fn, tp = confusion_matrix(test_predictions, test_data_target).ravel()
    # print ('tn fp')
    # print (tn,fp)
    # print('fn tp')
    # print(fn,tp)

    # print("TSVM Accuracy = ", accuracy)


    trn_labeled = train_l
    trn_unlabeled = train_unlabeled
    trn = np.concatenate((trn_labeled, trn_unlabeled), axis=0)

    data = np.concatenate((trn, test_data), axis=0)
    lin_ker = extension_cluster_kernel(data, 'linear')
    # eigenvalue
    eig = lin_ker.eigvalues

    cut_off = k_th_largest_eig(eig, 80)
    lin_ker.poly_step([cut_off,1/2,1])
 #   lin_ker.polynomial(9)
    clf_ck = svm.SVC(kernel=lin_ker.distance, C=C, class_weight='balanced')
    trn_target = np.array(train_targets)

    clf_ck.fit(trn_labeled, trn_target)
    accuracy_total[i, 2] = clf_ck.score(test_data, test_data_target)
    #
    #
    #
    #
    #
    #
    # Random walk
    unl_total = np.concatenate((train_unlabeled, test_data), axis = 0)
    ### Random Walk
    gam=0.00001   #0.00000001
    #prepei na kanw concatenate ta train unlabeled mazi me ta test data
    rw = random_walk(train_l, train_targets, unl_total, gam,80,2)   #2,4  #2,2  #20,2,0.00000001 # 25,2,0.00000001 # best 60,2,0.00000001
    res = rw.results
    predicted_res = res[(len(train_l)+len(train_unlabeled)):] # len(unlabeld) + len(labeled)
    test_target = np.array(test_data_target)
    accuracyRW = test_target[test_target==predicted_res].shape[0]/test_target.shape[0]
    accuracy_total[i, 3] = accuracyRW




# print('cluster_kernel',acc)
#  SVM, TSVM, Cluster kernel, Random Walk
print ('SVM, TSVM, CLUSTER KERNEL, RANDOM WALK')
print(accuracy_total.mean(axis=0))
print(accuracy_total.std(axis=0))