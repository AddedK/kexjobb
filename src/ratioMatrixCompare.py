import numpy as np
from collections import Counter

"""

EXTRACT PURE CLUSTERS FROM THE CLUSTERINGS
A PURE CLUSTER IS A CLUSTER LARGER THAN SIZE 1 WITH ONLY ONE NEURON TYPE
E.G. A CLUSTER WITH 5 PYRAMIDAL NEURONS IS CONSIDERED TO BE A PURE CLUSTER

"""

labels = np.loadtxt("dim_ap_labels.txt")
labels2 = np.loadtxt("dim_ward_labels.txt")

ratioMatrix1 = np.loadtxt("dim_ratioMatrixAp1.txt")
ratioMatrix2 = np.loadtxt("dim_ratioMatrixWard2.txt")

nrOfClusters = len(Counter(labels).keys())

pureClustersAp = []
pureClustersWard = []

label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]

for row, i in zip(ratioMatrix1, range(0, nrOfClusters)):
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersAp.append([i, name])

for row, i in zip(ratioMatrix2, range(0, nrOfClusters)):
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersWard.append([i, name])

pureClustersAp = np.asarray(pureClustersAp)
pureClustersWard = np.asarray(pureClustersWard)

np.savetxt("dim_apPureClusters.txt", pureClustersAp, fmt='%s')
np.savetxt("dim_wardPureClusters.txt", pureClustersWard, fmt='%s')
