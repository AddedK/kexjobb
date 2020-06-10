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

DIMENSIONALITY REDUCTION OF ORIGINAL DATA WITH PCA AND T-SNE
10 000 DIMENSIONS -> 50 PRINCIPAL COMPONENTS -> 2 DIMENSIONS WITH T-SNE

"""

#original file with all persistence images (one for each row)
X = np.loadtxt("X_train_all_types.txt")

feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feat_cols)
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# see how much the top three PCA components contribute
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]
print('Explained variation per principal component: {}'.format(
    pca.explained_variance_ratio_))

#extract top 50 principal components
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
print('Cumulative explained variation for 50 principal components: {}'.format(
    np.sum(pca_50.explained_variance_ratio_)))


#embed top 50 principal components into a 2-dimensional map with t-sne
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)

df['tsne-pca50-one'] = tsne_pca_results[:, 0]
df['tsne-pca50-two'] = tsne_pca_results[:, 1]

dim_reduced_data = np.column_stack(
    (tsne_pca_results[:, 0], tsne_pca_results[:, 1]))

np.savetxt("dim_reduced_data.txt", dim_reduced_data)







