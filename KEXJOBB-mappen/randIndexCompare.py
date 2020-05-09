import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

x = [1, 1, 2, 2, 3, 3, 3]
y = [2, 2, 3, 3, 1, 1, 1]

score = metrics.adjusted_rand_score(x, y)
print(score)


# labels = np.loadtxt("ap_labels.txt")
# print(labels)
# labels2 = np.loadtxt("ward_labels.txt")
# print(labels2)

# score = metrics.adjusted_rand_score(labels, labels2)

# print("rand score")
# print(score)
