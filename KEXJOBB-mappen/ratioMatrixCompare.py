import numpy as np

labels = np.loadtxt("ap_labels.txt")
labels2 = np.loadtxt("ward_labels.txt")

ratioMatrix1 = np.loadtxt("ratioMatrixAp1.txt")
ratioMatrix2 = np.loadtxt("ratioMatrixWard2.txt")

interestingClustersNumbersAp = []
interestingClustersNumbersWard = []

pureClustersAp = []
pureClustersWard = []

label_names = ["glutamatergic", "granule",
               "medium_spiny", "basket", "fast_spiking", "pyramidal"]


for row, i in zip(ratioMatrix1, range(0, 126)):
    if(np.any(row > 0.8)):
        # index = np.where(np.any(row > 0.8))
        index = np.where(row > 0.8)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        interestingClustersNumbersAp.append([i, name])
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersAp.append([i, name])

for row, i in zip(ratioMatrix2, range(0, 128)):
    if(np.any(row > 0.8)):
        index = np.where(row > 0.8)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        interestingClustersNumbersWard.append([i, name])
    if(np.any(row == 1.0)):
        index = np.where(row == 1.0)
        index = index[0]
        index = index[0]
        print(index)
        name = label_names[index]
        pureClustersWard.append([i, name])

interestingClustersNumbersAp = np.asarray(interestingClustersNumbersAp)
interestingClustersNumbersWard = np.asarray(interestingClustersNumbersWard)

pureClustersAp = np.asarray(pureClustersAp)
pureClustersWard = np.asarray(pureClustersWard)


np.savetxt("apAboveEightyClusters.txt", interestingClustersNumbersAp, fmt='%s')
np.savetxt("wardAboveEightyClusters.txt",
           interestingClustersNumbersWard, fmt='%s')
np.savetxt("apPureClusters", pureClustersAp, fmt='%s', delimiter=",")
np.savetxt("wardPureClusters", pureClustersWard, fmt='%s')
