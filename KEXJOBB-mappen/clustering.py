# Example to extract the persistence diagram from a neuronal tree

# Step 1: Import the tmd module
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


# type_name = "granule"
# saved_X_filename = type_name + '_X_train.txt'
# saved_X_filename = 'X_train_all_types.txt'
saved_X_filename = 'dim_reduced_data.txt'


X = np.loadtxt(saved_X_filename)
print(X.shape)


# st = time.time()
# ward_labels = AgglomerativeClustering(
#     n_clusters=3800, linkage='ward').fit_predict(X)
# elapsed_time = time.time() - st
# result = metrics.silhouette_score(X, ward_labels)

# print("WARD")
# print("Elapsed time: %.2fs" % elapsed_time)
# print("Number of points: %i" % ward_labels.size)
# # print(ward_labels)
# # print(*ward_labels, sep='\n')  # print full list without truncation
# print("max")
# print("silhouette score: {}".format(result))


# st = time.time()

# Affinity propagation finding out best silhouette score -----
dampingArray = [0.70]
maxIterArray = [5000]
convIterArray = [200]
preferenceArray = [-0.5]
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
                    resSamples = metrics.silhouette_samples(X, ap_labels)
                    print("done with iteration = {} ".format(i))
                except Exception as exception:
                    print("oops one condition didn't work!")
                    print("damp {0} maxiter {1} conviter{2}".format(
                        dampingValue, maxIterValue, convIterValue))
                    print(exception)
                    continue
                else:
                    continue


print("hard exiting cluster.py after clustering")
print(result)
sys.exit()
# print(max(resSamples))


# print(result)
# np.savetxt("sillhoueteSamples.txt", resSamples)
# print(*ap_labels, sep='\n')


nr_of_labels = [109, 452, 856, 459, 57, 1875]

# increment = 0
#  for x in nr_of_labels:
#         if i <= x + increment:
#             labelIndex = 0
#
#
#        increment += x


nr_clusters = len(Counter(ward_labels).keys())

resultMatrix = np.zeros((nr_clusters, 6))


i = 1
labelIndex = 0
for cluster in ward_labels:
    if i <= 109:
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

#  ------------

# ap_labels = AffinityPropagation(
#     max_iter=7000, damping=0.9, convergence_iter=100).fit_predict(X)
# elapsed_time = time.time() - st

# print("Affinity Propagation")
# # print("Elapsed time: %.2fs" % elapsed_time)
# # print("Number of points: %i" % ap_labels.size)

# print(*ap_labels, sep='\n')
# print("max")
# print(max(ap_labels))
