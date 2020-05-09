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


apRes = np.loadtxt("resultmatrix1.txt")
wardRes = np.loadtxt("resultmatrix2.txt")

apRatio = []
wardRatio = []

for row in apRes:
    apRatio.append(row/np.sum(row))

for row in wardRes:
    wardRatio.append(row/np.sum(row))

np.savetxt("ratioMatrixAp1.txt", apRatio)
np.savetxt("ratioMatrixWard2.txt", wardRatio)
