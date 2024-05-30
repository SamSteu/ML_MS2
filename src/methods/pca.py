import numpy as np
import matplotlib.pyplot as plt

## MS2

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
        self.exvar_ratio_vector = []


    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        self.mean = np.mean(training_data, axis=0)
        training_data = training_data - self.mean

        C = np.cov(training_data.T)
        eigvals, eigvecs = np.linalg.eigh(C)

        #sorting our eigvals and select the corresponding eigvecs
        eigvals = eigvals[:: -1]
        eigvecs = eigvecs[:, ::-1]

        self.W = eigvecs[:, :self.d]  # Dxd matrix
        eg = eigvals[:self.d]    # d values
        
        # Compute the explained variance
        exvar_ratio = np.sum(eg) / np.sum(eigvals) * 100

        self.exvar_ratio_vector = eg / np.sum(eg)
        
        return self.mean, self.W, exvar_ratio


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
        

    def plot_cum_explained_var(self):
        cumulative_explained_variance = np.cumsum(self.exvar_ratio_vector)

        # Plot cumulative explained variance ratio
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Cumulative Explained Variance Ratio by Principal Component')
        plt.axhline(y=0.9,color='gray',linestyle='--')
        plt.grid()
        plt.show()