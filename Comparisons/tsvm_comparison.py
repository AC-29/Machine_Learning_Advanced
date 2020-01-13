from paper_kernel import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import math
from sklearn import svm
from TSVM import *

def k_th_largest_eig(eig,k):
    arr = eig.copy()
    arr.sort() 
    return arr[-k]
def compute_accuracy(predictions, targets):
    """
    Computes the accuracy of predictions in relation to targets.
    """
    return predictions[predictions == targets].size / predictions.size
with open ('usbs_databases', 'rb') as fp:
    usps_file = pickle.load(fp) 

partition = usps_file[0]
df_usps_sample = usps_file[1]
df_usps_test = usps_file[2]

train = df_usps_sample.iloc[:,1:].to_numpy()
test = df_usps_test.iloc[:,1:].to_numpy()

targets_train = df_usps_sample.iloc[:,0].to_numpy()
targets_test = df_usps_test.iloc[:,0].to_numpy()

data = np.concatenate((train,test),axis=0)
data = np.concatenate((train,test),axis=0)

sigma = 6
gamma = 1/(2*(sigma**2))
C = 100
kernel = 'linear'
model = TSVM(kernel, gamma, C)

accuracyTSVM = np.zeros(49)
stdTSVM = np.zeros(49)
timeFitTSVM = np.zeros(49)
timeFitstdTSVM = np.zeros(49)
timePredTSVM = np.zeros(49)
timePredStdTSVM = np.zeros(49)





for i in range(50):
    if i ==0:
        data_partitionAux = partition[i].iloc[:,1:]
        targetsAux = partition[i].iloc[:,0].to_numpy()

    else:
        data_partitionAux = np.concatenate((data_partitionAux,partition[i].iloc[:,1:]))
        targetsAux = np.concatenate((targetsAux,partition[i].iloc[:,0].to_numpy()))





for iter in range(49):
    accuracy = np.zeros(math.floor(50/(iter+1)))
    timeFit = np.zeros(math.floor(50/(iter+1)))
    timePred = np.zeros(math.floor(50/(iter+1)))
    for i in range(math.floor(50/(iter+1))):
        data_partition = data_partitionAux[i*(40*(iter+1)):(i+1)*(40*(iter+1)),:]
        inputs =  data_partition
        targets = targetsAux[i*(40*(iter+1)):(i+1)*(40*(iter+1))]
        test_df = df_usps_test
        test_targets=test_df.iloc[:,0].to_numpy()
        test_inputs =test_df.iloc[:,1:].to_numpy()
        
        if i == 0:
            train_unlabeled = data_partitionAux[(i+1)*(40*(iter+1)):,:]
        
        elif i == 49:
            train_unlabeled = data_partitionAux[:i*(40*(iter+1)),:]
        
        else:
            train_unlabeled = np.concatenate((data_partitionAux[:i*(40*(iter+1)),:],data_partitionAux[(i+1)*(40*(iter+1)):,:]))
        
        
        
        #TSVM
        start = time.time()
        model.train(inputs, targets, train_unlabeled)
        end = time.time()
        timeFit[i] = end-start
        start = time.time()
        test_predictions = model.predict(test_inputs)
        end = time.time()
        timePred[i] = end-start
        accuracy[i] = compute_accuracy(test_predictions, test_targets)
        
        
        
        

        # clf = svm.SVC(kernel=ker.distance, C=100,class_weight='balanced')
        # start = time.time()
        # clf.fit(inputs,targets)
        # end = time.time()
        # timeFit[i] = end-start
        # start = time.time()
        # accuracy[i] = clf.score(test_inputs,test_targets)
        # end = time.time()
        # timePred[i] = end-start
        # #print('accuracy at iteration'+str(i)+':',accuracy[i])
        # print("sub-iteration: "+str(i)+" out of "+str(math.floor(50/(iter+1))))
        
    print("Iteration: ",iter)
    accuracyTSVM[iter] = np.mean(accuracy)
    stdTSVM[iter] = np.std(accuracy)
    timeFitTSVM[iter] = np.mean(timeFit)
    timeFitstdTSVM[iter] = np.std(timeFit)
    timePredTSVM[iter] = np.mean(timePred)
    timePredStdTSVM[iter] = np.std(timePred)
    

x = np.linspace(40,2000,50)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,accuracyTSVM,label = 'TSVM Accuracy')
plt.title('TSVM Accuracy with increaseing training samples')
# ax.legend()
plt.show()
ax = plt.subplot(111)
ax.plot(x,timeFitTSVM,'r',label = 'Fit Time')
ax.plot(x,timePredTSVM,'b',label = 'Predict Time')
plt.title('TSVM time elapsed with increasing training samples')
ax.legend()
plt.show()