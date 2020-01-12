from paper_kernel import extension_cluster_kernel
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

ker = extension_cluster_kernel(data,'linear')
eig = ker.eigvalues


accuracyCK = np.zeros(50)
stdCK = np.zeros(50)
timeFitCK = np.zeros(50)
timeFitstdCK = np.zeros(50)
timePredCK = np.zeros(50)
timePredStdCK = np.zeros(50)

def k_th_largest_eig(eig,k):
    arr = eig.copy()
    arr.sort() 
    return arr[-k]

cut_off = k_th_largest_eig(eig,15)
ker.poly_step([cut_off,1/2,2])


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
        clf = svm.SVC(kernel=ker.distance, C=100,class_weight='balanced')
        start = time.time()
        clf.fit(inputs,targets)
        end = time.time()
        timeFit[i] = end-start
        start = time.time()
        accuracy[i] = clf.score(test_inputs,test_targets)
        end = time.time()
        timePred[i] = end-start
        #print('accuracy at iteration'+str(i)+':',accuracy[i])
    accuracyCK[iter] = np.mean(accuracy)
    stdCK[iter] = np.std(accuracy)
    timeFitCK[iter] = np.mean(timeFit)
    timeFitstdCK[iter] = np.std(timeFit)
    timePredCK[iter] = np.mean(timePred)
    timePredStdCK[iter] = np.std(timePred)
    

x = np.linspace(0,50)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x,accuracyRVM,label = 'Cluster Kernel Accuracy')
plt.title('SVM Accuracy using Cluster Kernel with increaseing training samples')
ax.legend()
plt.show()
ax = plt.subplot(111)
ax.plot(x,timeFitRVM,'r',label = 'Fit Time')
ax.plot(x,timePredRVM,'b',label = 'Predict Time')
plt.title('SVM time elased using Cluster Kernel with increasing training samples')
ax.legend()
plt.show()