import sklearn_rvm as rvm
import numpy as np
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
import time
import math

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

#SVM
sigma=6
accuracySVM = np.zeros(50)
stdSVM = np.zeros(50)
timeFitSVM = np.zeros(50)
timeFitstdSVM = np.zeros(50)
timePredSVM = np.zeros(50)
timePredStdSVM = np.zeros(50)

gam = 1/(2*sigma**2)

for i in range(50):
    if i ==0:
        data_partitionAux = partition[i].iloc[:,1:]
        targetsAux = partition[i].iloc[:,0].to_numpy()
    else:
        data_partitionAux = np.concatenate((data_partitionAux,partition[i].iloc[:,1:]))
        targetsAux = np.concatenate((targetsAux,partition[i].iloc[:,0].to_numpy()))


for iter in range(50):
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
        clf = svm.SVC(kernel='rbf', C=100,gamma=gam,class_weight='balanced')
        start = time.time()
        clf.fit(inputs, targets)
        end = time.time()
        timeFit[i] = end-start
        start = time.time()
        accuracy[i] = clf.score(test_inputs,test_targets)
        end = time.time()
        timePred[i] = end-start
        #print('accuracy at iteration'+str(i)+':',accuracy[i])
    accuracySVM[iter] = np.mean(accuracy)
    stdSVM[iter] = np.std(accuracy)
    timeFitSVM[iter] = np.mean(timeFit)
    timeFitstdSVM[iter] = np.std(timeFit)
    timePredSVM[iter] = np.mean(timePred)
    timePredStdSVM[iter] = np.std(timePred)
    

x = np.linspace(0,50)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,accuracySVM,label = 'SVM Accuracy')
plt.title('SVM Accuracy with increaseing training samples')
ax.legend()
plt.show()
ax = plt.subplot(111)
ax.plot(x,timeFitSVM,'r',label = 'Fit Time')
ax.plot(x,timePredSVM,'b',label = 'Predict Time')
plt.title('SVM time elased with increasing training samples')
ax.legend()
plt.show()
