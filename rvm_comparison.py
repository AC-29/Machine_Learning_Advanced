import sklearn_rvm as rvm
import numpy as np
import pickle
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

#RVM
accuracyRVM = np.zeros(50)
stdRVM = np.zeros(50)
timeFitRVM = np.zeros(50)
timeFitstdRVM = np.zeros(50)
timePredRVM = np.zeros(50)
timePredStdRVM = np.zeros(50)


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
        rvc = rvm.RVC2(kernel = 'rbf')
        start = time.time()
        rvc.fit(inputs,targets)
        end = time.time()
        timeFit[i] = end-start
        start = time.time()
        accuracy[i] = rvc.score(test_inputs,test_targets)
        end = time.time()
        timePred[i] = end-start
        #print('accuracy at iteration'+str(i)+':',accuracy[i])
    accuracyRVM[iter] = np.mean(accuracy)
    stdRVM[iter] = np.std(accuracy)
    timeFitRVM[iter] = np.mean(timeFit)
    timeFitstdRVM[iter] = np.std(timeFit)
    timePredRVM[iter] = np.mean(timePred)
    timePredStdRVM[iter] = np.std(timePred)
    

x = np.linspace(0,50)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,accuracyRVM,label = 'RVM Accuracy')
plt.title('RVM Accuracy with increaseing training samples')
ax.legend()
plt.show()
ax = plt.subplot(111)
ax.plot(x,timeFitRVM,'r',label = 'Fit Time')
ax.plot(x,timePredRVM,'b',label = 'Predict Time')
plt.title('RVM time elased with increasing training samples')
ax.legend()
plt.show()
