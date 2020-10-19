from collections import defaultdict
from pprint import pprint

import gensim
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

model = gensim.models.KeyedVectors.load_word2vec_format('files/GoogleNews-vectors-negative300.bin', binary=True)

data = set()
whitelist = ['capital-common-countries', 'capital-world']
with open('files/questions-words.txt', encoding='utf-8') as f:
    key = ''
    for text in f.readlines():
        if text[0] == ':':
            key = text[1:].strip() if any([text[1:].strip() == x for x in whitelist]) else ''
        elif key:
            data.update(text.strip().split())

vectors = [model.wv[word] for word in sorted(list(data))]
kmeans_model = KMeans(n_clusters=5, random_state=42, n_jobs=-1).fit(vectors)

d = defaultdict(list)
for group, label in zip(kmeans_model.labels_, sorted(list(data))):
    d[group].append(label)

pprint(d)


def plot_dendrogram(model, **kwargs):
    """
    From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


ward = AgglomerativeClustering(linkage='ward', distance_threshold=0, n_clusters=None).fit(vectors)
plot_dendrogram(ward, truncate_mode='level', p=3)
plt.show()


reduced = TSNE(n_components=2, random_state=0).fit_transform(vectors)
plt.scatter(reduced[:, 0], reduced[:, 1])
plt.show()
