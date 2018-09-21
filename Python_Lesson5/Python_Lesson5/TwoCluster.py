import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
# load a dataset of height in  inches and weight in pounds of a person into an array
X = np.array( [[1.0,1.0], [1.5,2.0,],[3.0,4.0], [5.0,7.0], [3.5,5.0],[4.5,5.0] ] )
# plot them on the graph
plt.scatter(X[:,0],X[:,1], label='True Position')
plt.show()
# took 2 clusters for S, M size in tshirt and fit the data in X array
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
# show to which clusters they belong
print(kmeans.cluster_centers_)
# print the labels
print(kmeans.labels_)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
# show datapoints belonging to each centroid in different colors on the graph
plt.show()
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()
# show centroids on the graph in black color
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')
plt.show()