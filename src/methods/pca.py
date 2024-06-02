import numpy as np
import matplotlib.pyplot as plt

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        self.mean = None
        self.W = None
        self.exvar_ratio = None
        self.exvar_ratio_vector = []




    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        self.mean = np.mean(training_data, axis=0)
        training_data = training_data - self.mean
        C = np.cov(training_data.T)
        eigvals, eigvecs = np.linalg.eigh(C)

        # sorting our eigvals and select the corresponding eigvecs
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        self.W = eigvecs[:, :self.d]  # Dxd matrix
        eg = eigvals[:self.d]  # d values
        
        # Compute the explained variance and cumulative explained variance vector
        self.exvar_ratio = np.sum(eg) / np.sum(eigvals) * 100
        self.exvar_ratio_vector = np.cumsum(eigvals) / np.sum(eigvals) * 100
        
        return self.mean, self.W, self.exvar_ratio



    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        data = data - self.mean
        # project the data using W
        data_reduced = data @ self.W
        return data_reduced




    def reconstruct(self, data_reduced):
        """
        Reconstruct the data using the previously computed reduced data and the mean variance.

        Arguments:
            data_reduced (array): reduced data of shape (N,d)
        Returns:
            reconstructed_data (array): reconstructed data of shape (N,D)
        """
        reconstructed_data = self.mean + data_reduced @ self.W.T
        return reconstructed_data




    def plot_reconstruct_one_sample(self, data):
        # Choisir un échantillon aléatoire parmi les données
        np.random.seed(25)
        sample_id = np.random.randint(0, data.shape[0])
        sample_data = data[sample_id, :]

        sample_reduced_data = self.reduce_dimension(sample_data.reshape(1, -1))
        sample_reconstructed_data = self.reconstruct(sample_reduced_data)

        plt.figure(figsize=(8, 4))
        plt.suptitle(f'Using d={self.d} dimensions')

        ax = plt.subplot(1, 2, 1)
        plt.imshow(sample_data.reshape(28, 28), cmap='gray')
        ax.set_title('Original Image')

        ax = plt.subplot(1, 2, 2)
        plt.imshow(sample_reconstructed_data.reshape(28, 28), cmap='gray')
        ax.set_title('Reconstructed Image')
        plt.show()



    def plot_cum_explained_var(self):
        # Plot cumulative explained variance ratio
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.exvar_ratio_vector) + 1), self.exvar_ratio_vector, linestyle='solid')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio by Principal Component')
        plt.axhline(y=self.exvar_ratio, color='gray', linestyle='--')
        plt.axvline(x=self.d, color='gray', linestyle='--')
        plt.grid()
        plt.show()



    def plot_PCA_components(self):
        plt.figure(figsize=(8, 18))
        for i in range(10):
            plt.subplot(5, 2, i + 1)
            plt.imshow(self.W[:, i].reshape(28, 28), cmap='gray')
            plt.title(f'Principal Component: {i}')
        plt.show()
