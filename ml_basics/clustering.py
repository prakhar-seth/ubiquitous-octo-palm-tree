import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import cluster
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

x=np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]])

clf=KMeans(n_clusters=2)
clf.fit(x)
centroids=clf.cluster_centers_
labels=clf.labels_
colors=["g.","r.","c.","b.","k.","o."]
for i in range(len(x)):
    plt.plot(x[i][0],x[i][1],colors[labels[i]],markersize=10)
plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=100,linewidths=5)
plt.show()