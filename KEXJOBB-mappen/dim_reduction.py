from __future__ import print_function
from collections import Counter
from itertools import cycle
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics
import sys

"""
y = []
nr_of_labels = [109,452,856,459,57,1875]
for label_amount in nr_of_labels:
    for i in range(label_amount):
        y.append(label_amount)

y = np.asarray(y)

"""

X = np.loadtxt("X_train_all_types.txt")

print(X.shape)  # (70000, 784) (70000,)
# 3808*10000 för oss, samt 3808

feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
X, y = None, None
# Size of the dataframe: (70000, 785)
print('Size of the dataframe: {}'.format(df.shape))
# kommer vara 3808*10000 för oss


# see how much the top three PCA components contribute
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]
print('Explained variation per principal component: {}'.format(
    pca.explained_variance_ratio_))

"""
#kanske behöver ändra iom att vi har andra labels
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 126),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)

plt.show()

"""

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
print('Cumulative explained variation for 50 principal components: {}'.format(
    np.sum(pca_50.explained_variance_ratio_)))

sys.exit()

tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)

df['tsne-pca50-one'] = tsne_pca_results[:, 0]
df['tsne-pca50-two'] = tsne_pca_results[:, 1]


plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 126),
    # data=df.loc[rndperm, :],
    data=df,
    legend="full",
    alpha=0.3
)

plt.show()


sys.exit()

# ap_labels = AffinityPropagation(damping = 0.85, max_iter = 7000).fit_predict(np.column_stack((tsne_pca_results[:,0],tsne_pca_results[:,1])))

dim_reduced_data = np.column_stack(
    (tsne_pca_results[:, 0], tsne_pca_results[:, 1]))

np.savetxt("dim_reduced_data.txt", dim_reduced_data)

sys.exit()

print(dim_reduced_data.shape)

""" dampingArray = [0.70, 0.75, 0.85, 0.88, 0.90]
maxIterArray = [5000]
convIterArray = [20, 40, 60, 100, 150, 200]
preferenceArray = [-20, -5, -0.5, 0, 0.5, 5, 20]
answerArray = []


answerArray = []
i = 0

for dampingValue in dampingArray:
    for maxIterValue in maxIterArray:
        for convIterValue in convIterArray:
            for preferenceValue in preferenceArray:
                i = i + 1
                try:
                    af = AffinityPropagation(damping=dampingValue, preference=preferenceValue, max_iter=maxIterValue,
                                             convergence_iter=convIterValue).fit(dim_reduced_data)
                    cluster_centers_indices = af.cluster_centers_indices_
                    nrOfIterations = af.n_iter_
                    labels = af.labels_
                    number_of_clusters_found = max(labels)
                    euclScore = metrics.silhouette_score(
                        dim_reduced_data, labels, metric='euclidean')
                    manScore = metrics.silhouette_score(
                        dim_reduced_data, labels, metric='manhattan')
                    answerArray.append(
                        (dampingValue, maxIterValue, convIterValue, preferenceValue, euclScore, manScore))
                    print("done with iteration = {} ".format(i))
                except Exception as exception:
                    print("oops one condition didn't work!")
                    print("damp {0} maxiter {1} conviter{2} preferenceValue{3}  ".format(
                        dampingValue, maxIterValue, convIterValue, preferenceValue, number_of_clusters_found))
                    print(exception)
                    continue
                else:
                    continue

np.savetxt("dim_reduction_affinity2.txt", answerArray)
sys.exit()
 """
af = AffinityPropagation(damping=0.70, preference=-0.5, max_iter=2500,
                         convergence_iter=200).fit(dim_reduced_data)
cluster_centers_indices = af.cluster_centers_indices_

# ward = AgglomerativeClustering(n_clusters=677).fit(dim_reduced_data)
nrOfIterations = af.n_iter_
labels = af.labels_
number_of_clusters_found = max(labels)

# n_clusters_ = len(cluster_centers_indices)

# print("Number of iterations: %i " % nrOfIterations)
print("Number of points: %i" % labels.size)
print("Number of clusters : " + str(number_of_clusters_found))
print("Silhouette Coefficient euclidean: %0.3f"
      % metrics.silhouette_score(dim_reduced_data, labels, metric='euclidean'))
print("Silhouette Coefficient manhattan: %0.3f"
      % metrics.silhouette_score(dim_reduced_data, labels, metric='manhattan'))


np.savetxt("ap_dimreduced_labels.txt", labels)


sys.exit()

plt.close('all')
plt.figure(1)
plt.clf()

cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, n_clusters_))
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

"""
jet= plt.get_cmap('jet')
colors = iter(jet(np.linspace(0,1,52)))
colors = sns.color_palette(None, 52)
print(len(dim_reduced_data))

"""

"""

for k,col in zip(range(n_clusters_),colors):
    class_members = labels == k
    cluster_center = dim_reduced_data[cluster_centers_indices[k]]
    plt.plot(dim_reduced_data[class_members, 0], dim_reduced_data[class_members, 1], color = col )
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    for x in dim_reduced_data[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

"""


# [109,452,856,459,57,1875]
#["glutamatergic","granule", "medium_spiny", "basket", "fast_spiking", "pyramidal"]

idx = np.argsort(neuro_labels)
neuro_labels = np.array(neuro_labels)[idx]
labels = np.array(labels)[idx]

print("Basket results")
print(len(Counter(labels[0:459]).keys()))
print(Counter(labels[0:459]).keys())  # equals to list(set(words))
print(Counter(labels[0:459]).values())  # counts the elements' frequency

print("Fast Spiking results")
print(len(Counter(labels[459:516]).keys()))
print(Counter(labels[459:516]).keys())  # equals to list(set(words))
print(Counter(labels[459:516]).values())  # counts the elements' frequency

print("Glutamatergic results")
print(len(Counter(labels[516:625]).keys()))
print(Counter(labels[516:625]).keys())  # equals to list(set(words))
print(Counter(labels[516:625]).values())  # counts the elements' frequency

print("Granule results")
print(len(Counter(labels[625:1077]).keys()))
print(Counter(labels[625:1077]).keys())  # equals to list(set(words))
print(Counter(labels[625:1077]).values())  # counts the elements' frequency

print("Medium Spiny results")
print(len(Counter(labels[1077:1933]).keys()))
print(Counter(labels[1077:1933]).keys())  # equals to list(set(words))
print(Counter(labels[1077:1933]).values())  # counts the elements' frequency

print("Pyramidal results")
print(len(Counter(labels[1933:3808]).keys()))
print(Counter(labels[1933:3808]).keys())  # equals to list(set(words))
print(Counter(labels[1933:3808]).values())  # counts the elements' frequency

"""
print("Labels found")
for i in range(len(neuro_labels)):
    print(neuro_labels[i])
    print(labels[i])
    print("**************")
"""
