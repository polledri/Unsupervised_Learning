import numpy as np
import matplotlib.pyplot as plt
import sys

class VecteurGaussien:  
    def __init__(self, moyenne, matrice_covariance):
        self.moyenne = moyenne
        self.matrice_covariance = matrice_covariance

    def simulation(self, nb_simu):
        # Génération des échantillons selon la distribution gaussienne
        samples = np.random.multivariate_normal(self.moyenne, self.matrice_covariance, nb_simu)

        # Extraction des composantes X et Y des échantillons
        X = samples[:, 0]
        Y = samples[:, 1]

        # Plot des échantillons
        plt.scatter(X, Y)
        plt.title('Echantillons de la distribution gaussienne')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

        # Calcul de l'empirical average pour différentes tailles d'échantillons
        empirical_averages = []
        expected_value = np.array(self.moyenne)

        for i in range(1, nb_simu + 1):
            current_average = np.mean(samples[:i], axis=0)
            empirical_averages.append(np.linalg.norm(current_average - expected_value))

        # Plot de la convergence vers la valeur attendue
        plt.plot(range(1, nb_simu + 1), empirical_averages)
        plt.title('Convergence vers la valeur attendue')
        plt.xlabel('Taille de l\'échantillon')
        plt.ylabel('Distance euclidienne')
        plt.show()

if __name__ == "__main__":
    # Récupérer les arguments passés dans le terminal
    if len(sys.argv) != 6:
        print("Wrong nbr of args: mean_x, mean_y, cov_x, cov_y, nbr_simu")
        sys.exit(1)

    mean_x = int(sys.argv[1])
    mean_y = int(sys.argv[2])
    cov_x = int(sys.argv[3])
    cov_y = int(sys.argv[4])
    nbr_simu = int(sys.argv[5])

    gaussian_vector = VecteurGaussien([mean_x, mean_y], [[cov_x, 0], [0, cov_y]])
    gaussian_vector.simulation(nbr_simu)