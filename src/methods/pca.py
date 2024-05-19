import numpy as np

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
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None


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
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.mean = np.mean(training_data, axis=0)
        training_data = training_data - self.mean

        C = np.cov(training_data.T)
        eigvals, eigvecs = np.linalg.eigh(C)

        #sorting our eigvals and select the corresponding eigvecs
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        self.W = eigvecs[:, :self.d]  # Dxd matrix
        eg = eigvals[:self.d]    # d values
        
        # Compute the explained variance
        exvar = np.sum(eg) / np.sum(eigvals) * 100
        
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.find_principal_components(data)
        # project the data using W
        data_reduced = data @ self.W
        return data_reduced
        

