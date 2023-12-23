
class VecteurGaussien:import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class VecteurGaussien:  
    def __init__(self, moyenne, matrice_covariance):
        self.moyenne = moyenne
        self.matrice_covariance = matrice_covariance

    def simulation(self, nb_simu):
        # Génération des échantillons selon la distribution gaussienne
        samples = np.random.multivariate_normal(self.moyenne, self.matrice_covariance, nb_simu)
        # # Extraction des composantes X et Y des échantillons
        # X = samples[:, 0]
        # Y = samples[:, 1]
        self.samples=samples
    def plot_grah(self):
        # Plot des échantillons
        X = self.samples[:, 0]
        Y = self.samples[:, 1]
        plt.scatter(X, Y,label="Echantillions")
        plt.scatter(self.moyenne[0],self.moyenne[1],label="moyenne théorique")
        plt.title('Echantillons de la distribution gaussienne')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    
    def compute_empirical_mean(self,first_n:int=1,nb_simu:int=None):
        """Si nb_simu, on refait une simulation et on calcul la moyenne empirique
        sinon on utilise self.sample
        first_n permet de calculer la moyenne empirique sur le sous échantillion des first_n premier tirages"""
        # Calcul de l'empirical average pour différentes tailles d'échantillons
        if nb_simu:
            samples=self.simulation(nb_simu)
        else:
            if np.any(self.samples):
                samples=self.samples
            else:
                print("Pas de nombre de tirage pour la simu ni de donnée issu d'un tirage.")
                return(-1)

        empirical_averages = []
        expected_value = np.array(self.moyenne)

        for i in range(1, first_n+1):
            current_average = np.mean(samples[:i], axis=0)
            empirical_averages.append(np.linalg.norm(current_average - expected_value))

        # Plot de la convergence vers la valeur attendue
        plt.plot(range(1, first_n + 1), empirical_averages)
        plt.title('Convergence vers la valeur attendue')
        plt.xlabel('Taille de l\'échantillon')
        plt.ylabel('Norme 2 de l\'écart entre la valeur théorique \net empirique de la moyenne')
        plt.show()
