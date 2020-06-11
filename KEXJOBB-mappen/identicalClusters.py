"""

CALCULATES NUMBER OF IDENTICAL CLUSTERS

"""

import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os
from collections import Counter


rowCounts = []  # will contain n elements, where n is the nr of clusters

#for each cluster, contains all indexes for that cluster, so list of lists

apClusterIndexes = []
wardClusterIndexes = []

with open("dim_ap_cluster_indexes.txt") as ap_indexes:
    currentList = []
    for line in ap_indexes:
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            apClusterIndexes.append(currentList)
            currentList = []


with open("dim_ward_cluster_indexes.txt") as ward_indexes:
    currentList = []
    for line in ward_indexes:
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            wardClusterIndexes.append(currentList)
            currentList = []

rowCounts = [0] * 684

#rowCounts[i][1] means that two clusters agree

maxRatios = []
for i, ap_row in enumerate(apClusterIndexes):
    ratios = []
    for ward_row in wardClusterIndexes:
        if bool(set(ap_row).intersection(ward_row)):
            rowCounts[i] += 1
            #how similar are the two clusters compared?
            ratio = len(set(ap_row).intersection(ward_row)) / \
                (max(len(ap_row), len(ward_row)))
            ratios.append(ratio)
    #after comparing this cluster in ap to all clusters in ward, append the ratio between the most agreeing clusters
    #maxRatios is a list containing the highest similarity for all clusters in ap to some cluster in ward
    maxRatios.append(max(ratios))


count = 0
for i, ratio in enumerate(maxRatios):
    if ratio == 1:
        count += 1


print("COUNT : NUMBER OF IDENTICAL CLUSTERS")
print(count)

