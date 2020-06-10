import numpy as np
from collections import Counter

labels = np.loadtxt("dim_ap_labels.txt")
labels2 = np.loadtxt("dim_ward_labels.txt")

ratioMatrix1 = np.loadtxt("dim_ratioMatrixAp1.txt")
ratioMatrix2 = np.loadtxt("dim_ratioMatrixWard2.txt")

nrOfClusters = len(Counter(labels).keys())

interestingClustersNumbersAp = []
interestingClustersNumbersWard = []

pureClustersAp = []
pureClustersWard = []

label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]


for row, i in zip(ratioMatrix1, range(0, nrOfClusters)):
    if(np.any(row > 0.8)):
        # index = np.where(np.any(row > 0.8))
        index = np.where(row > 0.8)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        interestingClustersNumbersAp.append([i, name])
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersAp.append([i, name])

for row, i in zip(ratioMatrix2, range(0, nrOfClusters)):
    if(np.any(row > 0.8)):
        index = np.where(row > 0.8)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        interestingClustersNumbersWard.append([i, name])
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersWard.append([i, name])

interestingClustersNumbersAp = np.asarray(interestingClustersNumbersAp)
interestingClustersNumbersWard = np.asarray(interestingClustersNumbersWard)

pureClustersAp = np.asarray(pureClustersAp)
pureClustersWard = np.asarray(pureClustersWard)


np.savetxt("dim_apAboveEightyClusters.txt",
           interestingClustersNumbersAp, fmt='%s')
np.savetxt("dim_wardAboveEightyClusters.txt",
           interestingClustersNumbersWard, fmt='%s')
np.savetxt("dim_apPureClusters.txt", pureClustersAp, fmt='%s')
np.savetxt("dim_wardPureClusters.txt", pureClustersWard, fmt='%s')
