# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:38:43 2020

@author: Adrian
"""


import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def clusterKernel(dataTrain,dataUnlabeled,dataTest,k):
    
    #1 Compute affinity matrix, which is an RBF Kernel matrix but with
    #diagonal elements being 0 instead of 1.
    data = np.concatenate((dataTrain,dataUnlabeled,dataTest))
    affinityM = rbf_kernel(data)
    
    for i in range(affinityM.shape[0]):
        affinityM[i,i] = 0
                
    #2 Let D be a diagonal matrix with diagonal elemtns equal
    #to the sum of the rows (or the columns) of K and construct
    #the matrix L = D^(-1/2)*K*D^(-1/2). In this case, K (Kernel matrix)
    #will be affinityM
    D = np.zeros((affinityM.shape[0],affinityM.shape[1]))
    diagonalElements = np.sum(affinityM,axis = 1)
    
    for i in range(D.shape[0]):
        D[i,i] = diagonalElements[i]**(-0.5)
        
    aux = np.matmul(D,affinityM)
    L = np.matmul(aux,D)

    
    #3 Find the eigenvectors (v1,...,vk) of L corresponding the first k eigenvalues
    eValues,eVectors = np.linalg.eigh(L)
    eVectors = eVectors[:,0:k]
    
    #4 the new representation of the point xi is (vi1,....vik) and i snormalized
    #to have length one.
    newData = np.zeros((eVectors.shape[0],eVectors.shape[1]))
    for i in range(eVectors.shape[0]):
        denominator = np.sum([j**2 for j in eVectors[i,:]])
        newData[i,:] = [j/(denominator**(0.5)) for j in eVectors[i,:]]
        
    
    return newData
        
    
    
    
    
    
    
    
    
    
    