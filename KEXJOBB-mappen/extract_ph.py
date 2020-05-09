"""
# Import the TMD toolkit in IPython
import tmd

# Load a neuron
neuron = tmd.io.load_neuron('02a_pyramidal2aFI.CNG.swc')

# Extract the tmd of a neurite, i.e., neuronal tree
pd = tmd.methods.get_persistence_diagram(neuron.neurites[0])
"""

# Example to extract the persistence diagram from a neuronal tree

# Step 1: Import the tmd module
import sys
import csv
import numpy as np
import matplotlib.cm as cm
from tmd.view import view, plot
import tmd
import matplotlib.pyplot as plt
from tmd.view.common import jet_map
import os

# Step 2: Load your morphology
filename = '02a_pyramidal2aFI.CNG.swc'
neu = tmd.io.load_neuron(filename)

# Step 3: Extract the ph diagram of a tree
tree = neu.neurites[0]
ph = tmd.methods.get_persistence_diagram(tree)

# Step 4: Extract the ph diagram of a neuron's trees
ph_neu = tmd.methods.get_ph_neuron(neu)

# Step 5: Extract the ph diagram of a neuron's trees,
# depending on the neurite_type
ph_apical = tmd.methods.get_ph_neuron(neu, neurite_type='apical')
ph_axon = tmd.methods.get_ph_neuron(neu, neurite_type='axon')
ph_basal = tmd.methods.get_ph_neuron(neu, neurite_type='basal')

# Step 6: Plot the extracted topological data with three different ways


# Visualize the neuron
# view.neuron(neu)

# Visualize a selected neurite type or multiple of them
# view.neuron(neu, neurite_type=['apical'])

# Visualize the persistence diagram
# plot.diagram(ph_apical)

# Visualize the persistence barcode
# plot.barcode(ph_apical)

# Visualize the persistence image


# csv.field_size_limit(sys.maxsize)
# desired = [1, 3, 5]
# with open('X_train_all_types.txt', 'r') as fin:
#     reader = csv.reader(fin)
#     result = [[float(s) for s in row]
#               for i, row in enumerate(reader) if i in desired]

# print(np.array(result))

apPureClusterIndexes = []
with open("apPureClusterIndexes.txt") as f:
    currentList = []
    for line in f:
        print(line)
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            apPureClusterIndexes.append(currentList)
            currentList = []

apPureClusters = np.genfromtxt("apPureClusters.txt", usecols=1, dtype="str")


for i, indexes in enumerate(apPureClusterIndexes):
    path = "./pure_cluster_images/apPure/apPure{0}".format(i)
    os.mkdir(path)
    # print(indexes)
    for j, index in enumerate(indexes):
        result = np.loadtxt("X_train_all_types.txt",
                            skiprows=int(index)-1, max_rows=1)
        neuroMorphoName = apPureClusters[i]
        plt.imsave("{2}/apPureImage{0}{1}.png".format(neuroMorphoName, index, path),
                   result.reshape(100, 100), cmap=jet_map)

# This part of the code is to analyze the three largest clusters in both ap and ward

nr_of_labels = [109, 452, 856, 459, 57, 1875]
label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]


def getNeuroMorphoName(index):
    if index < 109:
        return label_names[0]
    elif index < 109+452:
        return label_names[1]
    elif index < 109+452+856:
        return label_names[2]
    elif index < 109+452+856+459:
        return label_names[3]
    elif index < 109+452+856+459+57:
        return label_names[4]
    else:
        return label_names[5]


# methodType = "ward"
# folderName = "./ward_third_largest_cluster_images"
# indexes = [524, 1458, 1528, 1749, 1936, 1975, 1996, 2015, 2018, 2039, 2040,
#            2042, 2056, 2065, 2076, 2080, 2092, 2102, 2129, 2165, 2200, 2214,
#            2218, 2255, 2265, 2283, 2337, 2360, 2366, 2396, 2406, 2413, 2420,
#            2445, 2479, 2480, 2481, 2505, 2574, 2605, 2615, 2633, 2646, 2667,
#            2679, 2688, 2707, 2718, 2734, 2748, 2752, 2755, 2796, 2807, 2808,
#            2813, 2823, 2825, 2827, 2839, 2843, 2852, 2854, 2862, 2882, 2891,
#            2947, 2962, 2985, 2995, 3002, 3033, 3049, 3059, 3082, 3085, 3104,
#            3116, 3132, 3136, 3151, 3176, 3187, 3189, 3204, 3215, 3223, 3227,
#            3247, 3249, 3258, 3262, 3268, 3272, 3305, 3313, 3388, 3429, 3468,
#            3476, 3485, 3515, 3561, 3562, 3623, 3626, 3630, 3632, 3664, 3671,
#            3682, 3747, 3761, 3783, 3792, 3797, 3802]

# for index in indexes:
#     result = np.loadtxt("X_train_all_types.txt",
#                         skiprows=int(index)-1, max_rows=1)
#     neuroMorphoName = getNeuroMorphoName(index)
#     plt.imsave("{2}/{3}_{0}_{1}.png".format(neuroMorphoName, index, folderName, methodType),
#                result.reshape(100, 100), cmap=jet_map)


# print("exiting from large cluster analysis")
# sys.exit()

# Testing pure clusters
# Ward

wardPureClusterIndexes = []
with open("wardPureClusterIndexes.txt") as f:
    currentList = []
    for line in f:
        print(line)
        if(line != "\n"):
            line = line[:-1]
            currentList.append(float(line))
        else:
            wardPureClusterIndexes.append(currentList)
            currentList = []

wardPureClusters = np.genfromtxt(
    "wardPureClusters.txt", usecols=1, dtype="str")


for i, indexes in enumerate(wardPureClusterIndexes):
    path = "./pure_cluster_images/wardPure/wardPure{0}".format(i)
    os.mkdir(path)
    # print(indexes)
    for j, index in enumerate(indexes):
        result = np.loadtxt("X_train_all_types.txt",
                            skiprows=int(index)-1, max_rows=1)
        neuroMorphoName = wardPureClusters[i]
        plt.imsave("{2}/wardPureImage{0}{1}.png".format(neuroMorphoName, index, path),
                   result.reshape(100, 100), cmap=jet_map)


# for i in result:
#     plt.imsave("testImage{i}.png".format(
#         i), np.array(i).reshape(100, 100), cmap=jet_map)
# plt.imshow(np.array(i).reshape(100, 100))


# print(ph_apical)
# plt.show(plot.persistence_image(ph_apical))
