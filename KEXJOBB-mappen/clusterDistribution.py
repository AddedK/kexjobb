# This is to see how many clusters agree with each other
import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os


rowCounts = []  # kommer innehålla 127 element


apClusterIndexes = []
wardClusterIndexes = []

with open("ap_cluster_indexes.txt") as ap_indexes:
    currentList = []
    for line in ap_indexes:
        # print(line)
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            apClusterIndexes.append(currentList)
            currentList = []


with open("ward_cluster_indexes.txt") as ward_indexes:
    currentList = []
    for line in ward_indexes:
        # print(line)
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            wardClusterIndexes.append(currentList)
            currentList = []

rowCounts = [0] * 127

# for i, ap_row in enumerate(apClusterIndexes):
#     for ward_row in wardClusterIndexes:
#         if bool(set(ap_row).intersection(ward_row)):
#             rowCounts[i] += 1

maxRatios = []
for i, ap_row in enumerate(apClusterIndexes):
    ratios = []
    for ward_row in wardClusterIndexes:
        if bool(set(ap_row).intersection(ward_row)):
            rowCounts[i] += 1
            ratio = len(set(ap_row).intersection(ward_row)) / \
                (max(len(ap_row), len(ward_row)))
            ratios.append(ratio)
    maxRatios.append(max(ratios))

# print(maxRatios)

count = 0
for i, ratio in enumerate(maxRatios):
    if ratio >= 0.5:
        count += 1
        print(i)
        print(ratio)
        print(apClusterIndexes[i])
print("COUNT")
print(count)


# for i, count in enumerate(rowCounts):
#     if count == 2:
#         print(i)
#         print(apClusterIndexes[i])
