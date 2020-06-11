
"""

RESULTMATRIX : EACH ROW CORRESPONDS TO A CLUSTER, HAS 6 COLUMNS WHICH REPRESENT EACH NEURON TYPE AND
HOW MANY OF EACH NEURON TYPE WERE CLUSTERED INTO THAT SPECIFIC CLUSTER

RATIOMATRIX : EACH ROW CORRESPONDS TO A CLUSTER, 6 COLUMNS, ILLUSTRATES THE DISTRIBUTION BETWEEN THE NEURON TYPES
FOR THAT CLUSTER INSTEAD OF THE AMOUNT

"""


import tmd
from tmd.view import view, plot
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
import time as time
from collections import Counter


apRes = np.loadtxt("dim_resultmatrix1.txt")
wardRes = np.loadtxt("dim_resultmatrix2.txt")

apRatio = []
wardRatio = []

for row in apRes:
    apRatio.append(row/np.sum(row))


for row in wardRes:
    wardRatio.append(row/np.sum(row))

np.savetxt("dim_ratioMatrixAp1.txt", apRatio)
np.savetxt("dim_ratioMatrixWard2.txt", wardRatio)
