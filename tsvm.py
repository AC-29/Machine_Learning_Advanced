import numpy as np
import sklearn.svm as svm

class TSVM:
    def __init__(self,kernel, gamma, C):
        self.Cl, self.Cu = 1.5, 0.001
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.clf = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma)

    def train(self, train_data, train_targets, train_unlabeled):

        N = len(train_data) + len(train_unlabeled)
        weight = np.ones(N)
        weight[len(train_data):] = self.Cu
        self.clf.fit(train_data, train_targets)
        train_unlabeled_prediction  = self.clf.predict(train_unlabeled)
        train_unlabeled_prediction  = np.expand_dims(train_unlabeled_prediction , 1)
        train_unlabeled_id = np.arange(len(train_unlabeled))
        train_targets = np.expand_dims(train_targets, 1)
        training_l_u = np.concatenate((train_data, train_unlabeled))
        training_targets_l_u = np.concatenate((train_targets, train_unlabeled_prediction ))

        while self.Cu < self.Cl:
            self.clf.fit(training_l_u, training_targets_l_u, sample_weight=weight)
            while True:
                train_unlabeled_prediction_d = self.clf.decision_function(train_unlabeled)
                train_unlabeled_prediction  = train_unlabeled_prediction .reshape(-1)
                margin = 1 - train_unlabeled_prediction  * train_unlabeled_prediction_d
                pos, pos_id = margin[train_unlabeled_prediction  > 0], train_unlabeled_id[train_unlabeled_prediction  > 0]
                neg, neg_id = margin[train_unlabeled_prediction  < 0], train_unlabeled_id[train_unlabeled_prediction  < 0]
                pos_id_max = pos_id[np.argmax(pos)]
                neg_id_max = neg_id[np.argmax(neg)]
                m, l = margin[pos_id_max], margin[neg_id_max]
                if m > 0 and l > 0 and m + l > 2.0:
                    train_unlabeled_prediction [pos_id_max] = -train_unlabeled_prediction [pos_id_max]
                    train_unlabeled_prediction [neg_id_max] = -train_unlabeled_prediction [neg_id_max]
                    train_unlabeled_prediction  = np.expand_dims(train_unlabeled_prediction , 1)
                    training_targets_l_u = np.concatenate((train_targets, train_unlabeled_prediction ))
                    self.clf.fit(training_l_u, training_targets_l_u, weight=weight)
                else:
                    break
            self.Cu = min(2*self.Cu, self.Cl)
            weight[len(train_data):] = self.Cu

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, Y):
        return self.clf.score(X, Y)
