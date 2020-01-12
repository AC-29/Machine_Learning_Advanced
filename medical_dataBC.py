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

gamma = 0.001


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
    def __init__(self, labeledData, labels, unlabeledData, gam=None, k=1, t=2):
        data = np.concatenate((labeledData, unlabeledData))
        L = labeledData.shape[0]
        N = data.shape[0]
        labels = (1 == labels) * 1
        self.W = rbf_kernel(data, gamma=gam)
        for i in range((self.W).shape[0]):
            sort = list(np.argsort(self.W[i]))
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
            number = (0.5 - np.random.rand()) * 0.4
            self.labelProbability[i, 0] = 0.5 + number
            self.labelProbability[i, 1] = 1 - self.labelProbability[i, 0]

        self.probability = np.zeros((N, L))
        oldloglike = -np.inf
        self.loglike = np.zeros(100)
        for iter in range(100):
            # E Step
            for i in range(N):
                for j in range(L):
                    self.probability[i, j] = self.labelProbability[i, labels[j]] * self.PT[i, j]

            # M Step
            for i in range(N):
                self.labelProbability[i, 0] = (np.sum((labels == 0) * self.probability[i])) / (
                    np.sum(self.probability[i]))
                self.labelProbability[i, 1] = (np.sum((labels == 1) * self.probability[i])) / (
                    np.sum(self.probability[i]))
            self.loglike[iter] = np.sum(np.log(np.sum(self.probability, axis=0)))
            if np.abs(self.loglike[iter] - oldloglike) < 10 ** (-4):
                break
            oldloglike = self.loglike[iter]

        self.posterior = np.matmul((self.labelProbability).transpose(), self.PT)
        self.results = ((np.argmax(self.posterior, axis=0)) * 2) - 1


dataset = load_breast_cancer()
dataset_features = np.array(dataset.data, dtype=float)
dataset_target = np.array(dataset.target, dtype=float)

for i in range(len(dataset_target)):
    if (dataset_target[i] == 0):
        dataset_target[i] = -1

print('number of -1:', dataset_target[dataset_target == -1].shape)
# Define train, unlabeled and test dataset

iterations = 10
accuracy_total = np.zeros((iterations, 3))
for i in range(iterations):
    np.random.seed(i)

    indeces = np.random.permutation(dataset_features.shape[0])
    # print(indeces)

    train_l = dataset_features[indeces[:40]]
    train_targets = dataset_target[indeces[0:40]]
    train_unlabeled = dataset_features[indeces[40:140]]
    train_unlabeled_target = dataset_target[indeces[40:140]]

    test_data = dataset_features[indeces[140:-1]]
    test_data_target = dataset_target[indeces[140:-1]]

    ######################################################

    ######### SVM
    clf = svm.SVC(C=100, gamma=gamma)
    clf.fit(train_l, train_targets)
    accuracy_total[i, 0] = clf.score(test_data, test_data_target)
    # print("SVM score = ", score)

    #######################################################

    ###########  TSVM
    C = 100
    kernel = 'linear'
    model = TSVM(kernel, gamma, C)
    model.train(train_l, train_targets, train_unlabeled)
    # print (train_target_[0:150])
    # Test TSVM
    test_predictions = model.predict(test_data)
    # performance
    accuracy_total[i, 1] = compute_accuracy(test_predictions, test_data_target)
    # print("TSVM Accuracy = ", accuracy)

    trn_labeled = train_l
    trn_unlabeled = train_unlabeled
    trn = np.concatenate((trn_labeled, trn_unlabeled), axis=0)

    data = np.concatenate((trn, test_data), axis=0)
    lin_ker = extension_cluster_kernel(data, 'linear')
    # eigenvalue
    eig = lin_ker.eigvalues

    cut_off = k_th_largest_eig(eig, i)
    # lin_ker.poly_step([cut_off,1/2,2])
    lin_ker.polynomial(9)
    clf_ck = svm.SVC(kernel=lin_ker.distance, C=100, class_weight='balanced')
    trn_target = np.array(train_targets)

    clf_ck.fit(trn_labeled, trn_target)
    accuracy_total[i, 2] = clf_ck.score(test_data, test_data_target)
    # print('cluster_kernel',acc)
#  SVM, TSVM, Cluster kernel
print(accuracy_total.mean(axis=0))
print(accuracy_total.std(axis=0))