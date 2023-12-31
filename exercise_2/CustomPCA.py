class CustomPCA:import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import numpy as np
class CustomPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(self.n_components)

    def simulation(self, data, labels):
        # Centrage et réduction des données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_reduced = self.pca.fit_transform(data_scaled)

        # Affichage des scatter plots en 2D ou 3D selon n_components
        if self.n_components == 2:
            plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=labels, cmap='viridis')
            plt.title('Reduction en 2D')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            legend = plt.colorbar()
            legend.set_label('Labels')
            plt.show()
        elif self.n_components == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2], c=labels, cmap='viridis')
            ax.set_title('Reduction en 3D')
            ax.set_xlabel('Composant 1')
            ax.set_ylabel('Composant 2')
            ax.set_zlabel('Composant 3')
            legend = plt.colorbar(scatter)
            legend.set_label('Labels')
            plt.show()
        else:
            print("Can only visualize in 2D or 3D.")
            sys.exit(1)

    def plot_eigen_value(self):
            """plot les valeurs propres de la matrice de covariance associée au jeu de données"""
            print(np.cumsum(self.pca.singular_values_))
            plt.plot([i for i in range(1,len(self.pca.singular_values_)+1)],self.pca.singular_values_)
            plt.title("Valeur propres de la matrice de covariance")
            plt.ylabel("vp")
            plt.xlabel("numéro vp")
            plt.show()
    def plot_explained_variance(self):
            """plot les valeurs propres de la matrice de covariance associée au jeu de données"""
            print(np.cumsum(self.pca.explained_variance_ratio_))
            plt.plot([i for i in range(1,len(self.pca.explained_variance_ratio_)+1)],np.cumsum(self.pca.explained_variance_ratio_))
            plt.title("Ratio de variance cumulé expliqué par chaques composantes")
            plt.ylabel("part de variance expliquée")
            plt.xlabel("Composante principale")
            plt.show()