"""

DOES CLUSTERING WITH AFFINITY PROPAGATION AND WARDS METHOD ON THE FINAL DATASET
DOES A PARAMETER TEST TO DETERMINE THE BEST SILHOUETTE SCORE 

"""
import tmd
from tmd.view import view, plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import sys
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import time as time
from collections import Counter

saved_X_filename = 'dim_reduced_data.txt'

X = np.loadtxt(saved_X_filename)



dampingArray = [0.70,0.75,0.85,0.88,0.90]
maxIterArray = [5000]
convIterArray = [40,60,100, 150,200]
preferenceArray = [-20, -5,-0.5, 0]
answerArray = []
i = 0

for dampingValue in dampingArray:
    for maxIterValue in maxIterArray:
        for convIterValue in convIterArray:
            for preferenceValue in preferenceArray:
                i = i + 1
                try:
                    # ward_labels = AgglomerativeClustering(
                    #     n_clusters=684).fit_predict(X)
                    ap_labels = AffinityPropagation(
                        max_iter=maxIterValue, damping=dampingValue, convergence_iter=convIterValue, preference=preferenceValue).fit_predict(X)
                    result = metrics.silhouette_score(
                        X, ap_labels)
                    answerArray.append(
                        (dampingValue, maxIterValue, convIterValue, preferenceValue, result))
                    print("done with iteration = {} ".format(i))
                except Exception as exception:
                    print("oops one condition didn't work!")
                    print("damp {0} maxiter {1} conviter{2}".format(
                        dampingValue, maxIterValue, convIterValue))
                    print(exception)
                    continue
                else:
                    continue



print(result)

neurons_per_type = [109, 452, 856, 459, 57, 1876]

nr_clusters = len(Counter(ward_labels).keys())

resultMatrix = np.zeros((nr_clusters, 6))

current_neuron_count = neurons_per_type[0]
current_neuron_count += current_neuron_count[i]

i = 1
labelIndex = 0
current_neuron_count = 0
for cluster in ward_labels:
    for index,nr_neurons in enumerate(neurons_per_type): 
        current_neuron_count += nr_neurons
        if i <= current_neuron_count:
            labelIndex = index
        resultMatrix[cluster][labelIndex] += 1
        i = i+1

i = 1
labelIndex = 0
for cluster in ward_labels:
    if i <= 109 : 
        labelIndex = 0
    elif i <= 109+452: 
        labelIndex = 1
    elif i <= 109+452+856:
        labelIndex = 2
    elif i <= 109+452+856+459:
        labelIndex = 3
    elif i <= 109+452+856+459+57:
        labelIndex = 4
    else:
        labelIndex = 5
    resultMatrix[cluster][labelIndex] += 1
    i = i+1

np.savetxt("dim_ward_labels.txt", ward_labels)
np.savetxt("dim_resultmatrix2.txt", resultMatrix)


