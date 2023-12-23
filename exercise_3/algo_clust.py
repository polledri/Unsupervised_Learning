import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN,OPTICS,cluster_optics_dbscan
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

class Clustering_supervise:
    """class qui effectue au choix un dbscan ou un optics avec a chaque fois 2 metrics de précision (calinski_harabasz_score et silhouette_score)"""
    def __init__(self,X):
        self.X=X

    def plot_distribution(self,dim1:int=0,dim2:int=1):
        plt.plot(self.X[:, dim1], self.X[:, dim2], 'o')
        plt.xlabel('dimension '+str(dim1))
        plt.ylabel('dimension '+str(dim2))
        plt.title("Visualisation des données sur un sous espace de dimension 2 choisit")
        plt.show()

    def dbscan_fit(self, eps=0.3, min_samples=10):
        Xcentered = StandardScaler().fit_transform(self.X)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(Xcentered)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        self.dbscan_labels = db.labels_

    def optics_fit(self,min_samples=50, xi=0.05, min_cluster_size=0.05):
        Xcentered = StandardScaler().fit_transform(self.X)
        clust = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size).fit(Xcentered)

        self.optics_labels_050 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=0.5,
        )

        self.optics_labels_200 = cluster_optics_dbscan(
        reachability=clust.reachability_,
        core_distances=clust.core_distances_,
        ordering=clust.ordering_,
        eps=2,
        )
    def compute_calinski_harabasz_score(self,db_or_optics:str='dbscan'):
        if db_or_optics=="optics":
            print("calinski_harabasz_score Optics 200: %0.3f"
      % metrics.calinski_harabasz_score(self.X, self.optics_labels_200))
            print("calinski_harabasz_score Optics 50: %0.3f"
      % metrics.calinski_harabasz_score(self.X, self.optics_labels_050))
        elif db_or_optics=="dbscan":
            print("calinski_harabasz_score dbscan: %0.3f"
      % metrics.calinski_harabasz_score(self.X, self.dbscan_labels))
        else:
            print("choissisez entre optics et dbscan")
            return -1

    def compute_silhouette_score(self,db_or_optics:str='dbscan'):
        if db_or_optics=="optics":
            print("Silhouette Coefficient Optics 200: %0.3f"
      % metrics.silhouette_score(self.X, self.optics_labels_200))
            print("Silhouette Coefficient Optics 50: %0.3f"
      % metrics.silhouette_score(self.X, self.optics_labels_050))
        elif db_or_optics=="dbscan":
            print("Silhouette Coefficient dbscan: %0.3f"
      % metrics.silhouette_score(self.X, self.dbscan_labels))
        else:
            print("choissisez entre optics et dbscan")
            return -1