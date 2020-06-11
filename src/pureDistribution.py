import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os


"""

CHECKS HOW MANY PURE CLUSTERS THERE ARE FOR EACH NEURON TYPE

"""

wardPureClusters = np.genfromtxt(
    "dim_wardPureClusters.txt", usecols=1, dtype="str")


wardPureClusterIndexes = []
with open("dim_wardPureClusterIndexes.txt") as f:
    currentList = []
    for line in f:
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            wardPureClusterIndexes.append(currentList)
            currentList = []


label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]

typeDistribution = [0] * 6
for i, cluster in enumerate(wardPureClusterIndexes):
    if len(cluster) > 1:
        neuroMorphoName = wardPureClusters[i]
        index = label_names.index(neuroMorphoName)
        typeDistribution[index] += 1


print(typeDistribution)
