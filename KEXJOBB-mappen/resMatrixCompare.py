from collections import Counter
import numpy as np
import sys

labels = np.loadtxt("ap_labels.txt")
labels2 = np.loadtxt("ward_labels.txt")

resultMatrix1 = np.loadtxt("resultmatrix1.txt")
resultMatrix2 = np.loadtxt("resultmatrix2.txt")


"""

Find largest cluster from method 1

"""


# cluster numbers
cluster_numbers1 = Counter(labels).keys()

# n_clusters_method_1 =  len(cluster_numbers1)
# Number in each cluster
amount_in_each_cluster1 = Counter(labels).values()

cluster_numbers1 = list(cluster_numbers1)
amount_in_each_cluster1 = list(amount_in_each_cluster1)

# simultaneous sort of two lists
print(len(amount_in_each_cluster1))
print(cluster_numbers1)


idx1 = np.argsort(amount_in_each_cluster1)
amount_in_each_cluster1 = np.array(amount_in_each_cluster1)[idx1]
sorted_cluster_numbers1 = np.array(cluster_numbers1)[idx1]


# print(int(sorted_cluster_numbers1[0]))
largest_cluster_nr1 = resultMatrix1[int(sorted_cluster_numbers1[126])]
second_largest_cluster_nr1 = resultMatrix1[int(sorted_cluster_numbers1[125])]
third_largest_cluster_nr1 = resultMatrix1[int(sorted_cluster_numbers1[124])]


"""

Find largest cluster from method 2

"""

# cluster numbers
cluster_numbers2 = Counter(labels2).keys()

# n_clusters_method_2 =  len(cluster_numbers2)

# Number in each cluster
amount_in_each_cluster2 = Counter(labels2).values()

cluster_numbers2 = list(cluster_numbers2)
amount_in_each_cluster2 = list(amount_in_each_cluster2)

# simultaneous sort of two lists
idx2 = np.argsort(amount_in_each_cluster2)
amount_in_each_cluster2 = np.array(amount_in_each_cluster2)[idx2]
sorted_cluster_numbers2 = np.array(cluster_numbers2)[idx2]
print("cluster_numbers2")
print(sorted_cluster_numbers2)


with open("ap_cluster_indexes.txt", "wb") as f:
    for cluster_nr in range(127):
        ii1 = np.where(labels == cluster_nr)[0]
        np.savetxt(f, ii1, fmt='%.5f')
        f.write(b'\n')

with open("ward_cluster_indexes.txt", "wb") as f:
    for cluster_nr in range(127):
        ii1 = np.where(labels2 == cluster_nr)[0]
        np.savetxt(f, ii1, fmt='%.5f')
        f.write(b'\n')


print("ending after making cluster indexes")
sys.exit()

print("shouldn't get here")

# largest_cluster_nr2 = resultMatrix2[int(cluster_numbers[126])]
# second_largest_cluster_nr2 = resultMatrix2[int(cluster_numbers[125])]
# third_largest_cluster_nr2 = resultMatrix2[int(cluster_numbers[124])]

"""

Create resultlists for mapping from clusters to neuromorpholabels

"""

nr_of_labels = [109, 452, 856, 459, 57, 1875]
label_names = ["glutamatergic", "granule", "medium_spiny",
               "basket", "fast_spiking", "pyramidal"]


# [][][][][][]
# will give us the 6 neuromorpho labels lists for the largest cluster from both bethods
# we want to compare these


for i in range(126, -1, -1):
    print("ap {0} {1} original clusterNr: {2} ".format(
        i, resultMatrix1[int(sorted_cluster_numbers1[i])], sorted_cluster_numbers1[i]))
    print("ward {0} {1} original clusterNr: {2}  ".format(
        i, resultMatrix2[int(sorted_cluster_numbers2[i])], sorted_cluster_numbers2[i]))
    print("\n")

print("indexes for largest clusters in ap")
searchval = sorted_cluster_numbers1[126]
ii1 = np.where(labels == searchval)[0]
print(repr(ii1))
print(len(ii1))

print("indexes for largest clusters in ward")
searchval = sorted_cluster_numbers2[126]
ii2 = np.where(labels2 == searchval)[0]
print(repr(ii2))
print(len(ii2))


print("indexes for 2nd largest clusters in ap")
searchval = sorted_cluster_numbers1[125]
ii1 = np.where(labels == searchval)[0]
print(repr(ii1))
print(len(ii1))

print("indexes for 2nd largest clusters in ward")

searchval = sorted_cluster_numbers2[125]
ii2 = np.where(labels2 == searchval)[0]
print(repr(ii2))
print(len(ii2))

print("indexes for 3rd largest clusters in ap")
searchval = sorted_cluster_numbers1[124]
ii1 = np.where(labels == searchval)[0]
print(repr(ii1))
print(len(ii1))

print("indexes for 3rd largest clusters in ward")
searchval = sorted_cluster_numbers2[124]
ii2 = np.where(labels2 == searchval)[0]
print(repr(ii2))
print(len(ii2))

# print("exiting")
# sys.exit()

apPureClusters = np.loadtxt("apPureClusters.txt", usecols=0)
wardPureClusters = np.loadtxt("wardPureClusters.txt", usecols=0)


apPureIndexes = []
wardPureIndexes = []
# print("indexes for pure clusters in both methods")

with open("apPureClusterIndexes.txt", "wb") as f:
    for row in apPureClusters:
        searchval = row
        ii1 = np.where(labels == searchval)[0]
        np.savetxt(f, ii1, fmt='%.5f')
        f.write(b'\n')
        # apPureIndexes.append(ii1)
        # print(repr(ii1))
        # print(len(ii1))


with open("wardPureClusterIndexes.txt", "wb") as f:
    for row in wardPureClusters:
        searchval = row
        ii2 = np.where(labels == searchval)[0]
        np.savetxt(f, ii2, fmt='%.5f')
        f.write(b'\n')
        # apPureIndexes.append(ii1)
        # print(repr(ii1))
        # print(len(ii1))


# for row in wardPureClusters:
#     searchval = row
#     ii2 = np.where(labels2 == searchval)[0]
#     wardPureIndexes.append(ii2)
#     # print(repr(ii2))
#     # print(len(ii2))

# np.savetxt("wardPureClusterIndexes.txt",
#            wardPureIndexes,  fmt='%s', delimiter=",")


"""

1 test

AP vs WARD 126 clusters without dimensionality reduction

"""

# len(Counter(labels).keys())


"""

"""
