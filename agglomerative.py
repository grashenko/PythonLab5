import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

df_raw = pd.read_json(r"data.json")
data = df_raw[['age','countGroups']]


X = np.array(data)

X = np.delete(X, np.where(X == 0)[0], axis=0)

fig = plt.figure(figsize=(5, 5))

dendrogram = sch.dendrogram(sch.linkage(X, method='average'))

plt.show(block = False)

fig = plt.figure(figsize=(5, 5))

model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='average')
model.fit(X)

labels = model.labels_

print(labels)

plt.scatter(X[labels==0, 0], X[labels==0, 1], s=50, marker='o', color='red')
plt.scatter(X[labels==1, 0], X[labels==1, 1], s=50, marker='o', color='blue')

plt.show(block = False)

fig = plt.figure(figsize=(5, 5))


df_raw = pd.read_json(r"data.json")
data = df_raw[['age','countGroups', 'cityId']]

X = np.array(data)

dendrogram = sch.dendrogram(sch.linkage(X, method='average'))

plt.show()

