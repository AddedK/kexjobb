"""

CALCULATES NUMBER OF CLUSTERS PER NEURON TYPE FOR BOTH METHODS

"""


import numpy as np
from collections import Counter

ap_labels = np.loadtxt("dim_ap_labels.txt")
ward_labels = np.loadtxt("dim_ward_labels.txt")


nr_of_labels = [109, 452, 856, 459, 57, 1875]
label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]


nr_of_clusters_ap = []
nr_of_clusters_ward = []

start = 0
end = nr_of_labels[0]

for i in range(len(nr_of_labels)):
    if(start != 0):
        end += nr_of_labels[i]
    sublist_ap = ap_labels[start:end]
    sublist_ward = ward_labels[start:end]
    nr_of_clusters_ap.append(len(Counter(sublist_ap).keys()))
    nr_of_clusters_ward.append(len(Counter(sublist_ward).keys()))

    start = end

print("AP")
print(nr_of_clusters_ap)
print("WARD")
print(nr_of_clusters_ward)
