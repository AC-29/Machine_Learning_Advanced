import csv
import numpy as np
from sklearn import svm
from TSVM import *
from paper_kernel import *
import pandas as pd
from sklearn.metrics import f1_score

with open('CTG_data.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    headers = next(reader)
    data = np.array(list(reader)).astype(float)
def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size
def k_th_largest_eig(eig,k):
    arr = eig.copy()
    arr.sort() 
    return arr[-k]

pand = pd.DataFrame(data)
pand = pand.drop_duplicates()
data = pand.to_numpy()

plus = data[:,:1645]
minus = data[:,1645:]
targets=data[:,-1]
data=data[:,:-1]

sigma = 6
gamma = 1/(2*(sigma**2))

trainSize = 40
iterations = 50
accuracy_total = np.zeros((iterations, 4))
count = 0
for i in range(iterations):
    #
    lengthTrain = 1
    
    while lengthTrain == 1:
        # np.random.seed(i+count)
        indeces = np.random.permutation(data.shape[0])
        train_targets = targets[indeces[0:trainSize]]
        lengthTrain = len(np.unique(train_targets))
        # count += count
        
    train_l = data[indeces[:trainSize]]    
    train_unlabeled = data[indeces[trainSize:(trainSize+100)]]
    train_unlabeled_target = targets[indeces[trainSize:(trainSize+100)]]
    test_data = data[indeces[(trainSize+100):-1]]
    test_data_target = targets[indeces[(trainSize+100):-1]]
        

    # ## SVM ##
    clf = svm.SVC(C=100, gamma=gamma,kernel = 'rbf')
    clf.fit(train_l, train_targets)
    prediction = clf.predict(test_data)
    f1SVM = f1_score(test_data_target,prediction,average = 'weighted')
    accuracy_total[i, 0] = clf.score(test_data, test_data_target)
    print("UNIQUE SVM: ")
    print(np.unique(prediction))
    
    
    # ## TSVM ##
    C = 100
    kernel = 'rbf'
    model = TSVM(kernel,gamma,C)
    model.train(train_l, train_targets, train_unlabeled)
    test_predictions = model.predict(test_data)
    f1TSVM = f1_score(test_data_target,test_predictions,average = 'weighted')
    accuracy_total[i, 1] = compute_accuracy(test_predictions, test_data_target)
    print("UNIQUE TSVM: ")
    print(np.unique(test_predictions))
    
    trn_labeled = train_l
    trn_unlabeled = train_unlabeled
    trn = np.concatenate((trn_labeled, trn_unlabeled), axis=0)
    
    # Extension Cluster Kernel ##
    dataAux = np.concatenate((trn, test_data), axis=0)
    kernel = kernel_function('rbf',gamma = gamma)
    lin_ker = extension_cluster_kernel(dataAux, kernel,'polynomial',12)
    eig = lin_ker.eigvalues
    trn_target = np.array(train_targets)
    # JH = JH_bound(trn_labeled,trn_target,test_data,gamma)
    # print(JH)
    # cut_off = k_th_largest_eig(eig, 10)
    # lin_ker.poly_step([cut_off,1/2,1])
    # lin_ker.step(cut_off)
    lin_ker.polynomial(12)
    clf_ck = svm.SVC(kernel=lin_ker.distance, C=100, class_weight='balanced')
    
    clf_ck.fit(trn_labeled, trn_target)
    prediction = clf.predict(test_data)
    f1CK = f1_score(test_data_target,prediction,average = 'weighted')
    accuracy_total[i, 2] = clf_ck.score(test_data, test_data_target)
    print("UNIQUE CK: ")
    print(np.unique(prediction))
    print(accuracy_total[i,2])
    
    
    # #RW
    unlabeled_data = np.concatenate((train_unlabeled,test_data))
    unlabeled_targets = np.concatenate((train_unlabeled_target,test_data_target))
    
    rw = random_walk(train_l,train_targets,unlabeled_data,gam = 0.027, k = 5, t = 4)
    accuracy_total[i,3] = np.sum(1*(rw.results[len(train_targets):] == unlabeled_targets))/len(unlabeled_targets)
    prediction = rw.results
    labels_all = np.concatenate((train_targets,test_data_target))
    f1RW = f1_score(labels_all,prediction,average = 'weighted')
    print("UNIQUE RW: ")
    print(np.unique(rw.results))
    # aux = 0
    # for j in range(len(train_targets)):
    #     aux+=(rw.results[len(train_targets)+j] ==int(unlabeled_targets[j]))*1
    
    # accuracy_total[i,3] = aux/len(unlabeled_targets)
    print(accuracy_total[i,3])
    
    
    
print(accuracy_total.mean(axis=0))
print(accuracy_total.std(axis=0))
    
    
    
    
    
    
# labelsNormal = labels[:1654]
# labelsPathological = labels[1654:]

# normal=data[:1654]
# pathological = data[1654:]
# inputs = np.concatenate((normal[:int((trainingData/2))],pathological[:int((trainingData/2))]))
# targets=np.concatenate((labelsNormal[:int((trainingData/2))],labelsPathological[:int((trainingData/2))]))
# test = np.concatenate((normal[int((trainingData/2)):],pathological[int((trainingData/2)):]))
# targetsTest = np.concatenate((labelsNormal[int((trainingData/2)):],labelsPathological[int((trainingData/2)):]))
# clf = svm.SVC(kernel='rbf', C=100,class_weight='balanced')
# clf.fit(inputs,targets)
