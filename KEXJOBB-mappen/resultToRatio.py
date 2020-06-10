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
