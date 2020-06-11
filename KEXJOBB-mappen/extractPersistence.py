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

LOAD PURE CLUSTER IMAGES 

"""

def load_pure_cluster_images():

	#load the pure cluster indexes into a list

	apPureClusterIndexes = []
	with open("dim_apPureClusterIndexes.txt") as f:
	    currentList = []
	    for line in f:
	        print(line)
	        if(line != "\n"):
	            line = line[:-1]
	            currentList.append(float(line))
	        else:
	            apPureClusterIndexes.append(currentList)
	            currentList = []

	apPureClusters = np.genfromtxt(
	    "dim_apPureClusters.txt", usecols=1, dtype="str")

	#for each neuron in each pure cluster, extract the persistence image into a png image

	for i, indexes in enumerate(apPureClusterIndexes):
	    path = "./dimImages/dim_pure_images/dim_ap_pure_images/dim_apPure{0}".format(
	        i)
	    os.mkdir(path)
	    # print(indexes)
	    for j, index in enumerate(indexes):
	        result = np.loadtxt("X_train_all_types.txt",
	                            skiprows=int(index)-1, max_rows=1)
	        neuroMorphoName = apPureClusters[i]
	        plt.imsave("{2}/dim_apPureImage{0}{1}.png".format(neuroMorphoName, index, path),
	                   result.reshape(100, 100), cmap=jet_map)

	#load the pure cluster indexes into a list

	wardPureClusterIndexes = []
	with open("dim_wardPureClusterIndexes.txt") as f:
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
	    "dim_wardPureClusters.txt", usecols=1, dtype="str")

	#for each neuron in each pure cluster, extract the persistence image into a png image

	for i, indexes in enumerate(wardPureClusterIndexes):
	    path = "./dimImages/dim_pure_images/dim_ward_pure_images/dim_wardPure{0}".format(
	        i)
	    os.mkdir(path)
	    # print(indexes)
	    for j, index in enumerate(indexes):
	        result = np.loadtxt("X_train_all_types.txt",
	                            skiprows=int(index)-1, max_rows=1)
	        neuroMorphoName = wardPureClusters[i]
	        plt.imsave("{2}/dim_wardPureImage{0}{1}.png".format(neuroMorphoName, index, path),
	                   result.reshape(100, 100), cmap=jet_map)





#load_pure_cluster_images()

