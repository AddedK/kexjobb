import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation

ap_labels = np.loadtxt("dim_ap_labels.txt")
ward_labels = np.loadtxt("dim_ward_labels.txt")


score = metrics.adjusted_rand_score(ap_labels, ward_labels)
print(score)
