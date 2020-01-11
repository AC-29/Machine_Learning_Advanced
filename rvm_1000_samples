#in order to have sklearn_rvm it is necessary to install rvm as: pip install sklearn-rvm

import sklearn_rvm as rvm
import numpy as np
import pickle
with open ('usbs_databases', 'rb') as fp:
    usps_file = pickle.load(fp) 

partition = usps_file[0]
df_usps_sample = usps_file[1]
df_usps_test = usps_file[2]
trn = df_usps_sample.iloc[:,1:].to_numpy()
test = df_usps_test.iloc[:,1:].to_numpy()
targets_train = df_usps_sample.iloc[:,0].to_numpy()
targets_test = df_usps_test.iloc[:,0].to_numpy()
data = np.concatenate((trn,test),axis=0)
accuracy = np.zeros(50)
# for i in range(50):
#     data_partition = partition[i].iloc[:,1:]
#     targets = partition[i].iloc[:,0].to_numpy()
#     inputs =  data_partition.to_numpy()
#     unlabeled_df = df_usps_sample.drop(partition[i].index) #'drop' simply ingores the indeces from particion [i], giving us the 'rest'
#     unlabeled_targets=unlabeled_df.iloc[:,0].to_numpy()
#     unlabeled_inputs =unlabeled_df.iloc[:,1:].to_numpy()
#     test_df = df_usps_test
#     test_inputs = test_df.iloc[:,1:].to_numpy()
#     test_targets= test_df.iloc[:,0].to_numpy()
#     rvc = rvm.RVC2(kernel='rbf')
#     rvc.fit(inputs,targets)
#     accuracy[i] = rvc.score(test_inputs,test_targets)
#     print(accuracy[i])


for i in range(25):
    if i ==0:
        data_partition = partition[i].iloc[:,1:]
        targets = partition[i].iloc[:,0].to_numpy()
        inputs =  data_partition.to_numpy()
        # unlabeled_df = df_usps_sample.drop(partition[i].index)
        # unlabeled_targets=unlabeled_df.iloc[:,0].to_numpy()
        # unlabeled_inputs =unlabeled_df.iloc[:,1:].to_numpy()
        test_df = df_usps_test
        test_inputs = test_df.iloc[:,1:].to_numpy()
        test_targets= test_df.iloc[:,0].to_numpy()
    else:
        data_partition = partition[i].iloc[:,1:]
        auxTargets = partition[i].iloc[:,0].to_numpy()
        targets = np.concatenate((targets,auxTargets))
        auxInputs =  data_partition.to_numpy()
        inputs = np.concatenate((inputs,auxInputs))
        # unlabeled_df = df_usps_sample.drop(partition[i].index)
        # auxUnlabeledTargets=unlabeled_df.iloc[:,0].to_numpy()
        # unlabeled_targets = np.concatenate((unlabeled_targets,auxUnlabeledTargets))
        # auxUnlabeledInputs = unlabeled_df.iloc[:,1:].to_numpy()
        # unlabeled_inputs = np.concatenate((unlabeled_inputs,auxUnlabeledInputs))
        auxtest_df = df_usps_test
        auxtest_inputs = test_df.iloc[:,1:].to_numpy()
        test_inputs = np.concatenate((test_inputs,auxtest_inputs))
        auxtest_targets= test_df.iloc[:,0].to_numpy()
        test_targets= np.concatenate((test_targets,auxtest_targets))
        
    
rvc = rvm.RVC2(kernel='rbf')
rvc.fit(inputs,targets)
accuracy = rvc.score(test_inputs,test_targets)
print("Accuracy 1: ",accuracy)



for i in range(25,50):
    if i ==25:
        data_partition = partition[i].iloc[:,1:]
        targets = partition[i].iloc[:,0].to_numpy()
        inputs =  data_partition.to_numpy()
        # unlabeled_df = df_usps_sample.drop(partition[i].index)
        # unlabeled_targets=unlabeled_df.iloc[:,0].to_numpy()
        # unlabeled_inputs =unlabeled_df.iloc[:,1:].to_numpy()
        test_df = df_usps_test
        test_inputs = test_df.iloc[:,1:].to_numpy()
        test_targets= test_df.iloc[:,0].to_numpy()
    else:
        data_partition = partition[i].iloc[:,1:]
        auxTargets = partition[i].iloc[:,0].to_numpy()
        targets = np.concatenate((targets,auxTargets))
        auxInputs =  data_partition.to_numpy()
        inputs = np.concatenate((inputs,auxInputs))
        # unlabeled_df = df_usps_sample.drop(partition[i].index)
        # auxUnlabeledTargets=unlabeled_df.iloc[:,0].to_numpy()
        # unlabeled_targets = np.concatenate((unlabeled_targets,auxUnlabeledTargets))
        # auxUnlabeledInputs = unlabeled_df.iloc[:,1:].to_numpy()
        # unlabeled_inputs = np.concatenate((unlabeled_inputs,auxUnlabeledInputs))
        auxtest_df = df_usps_test
        auxtest_inputs = test_df.iloc[:,1:].to_numpy()
        test_inputs = np.concatenate((test_inputs,auxtest_inputs))
        auxtest_targets= test_df.iloc[:,0].to_numpy()
        test_targets= np.concatenate((test_targets,auxtest_targets))

rvc = rvm.RVC2(kernel='rbf')
rvc.fit(inputs,targets)
accuracy = rvc.score(test_inputs,test_targets)
print("Accuracy 2: ",accuracy)
