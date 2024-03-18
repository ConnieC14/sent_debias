import numpy as np

class PCA_():
    def __init__(
        self,
        n_components=None
    ):
        self.n_components = n_components
    
    def fit(self, X):
        # Standardize the range of continuous initial variables
        mean = np.mean(X, axis=0)
        std = np.std(X)
        epsilon = 1e-7
        norm_X = (X - mean) / (std + epsilon)

        # Compute covariance matrix to identify correlations
        cov_X = np.dot(norm_X.T, norm_X) / len(X.shape[0] - 1)

        # Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
        eig_values, eig_vectors = np.linalg.eig(cov_X)

        # Create a feature vector to decide which principal components to keep
        idx_vals = np.argsort(eig_values)[::-1]
        feat_vect = eig_vectors[idx_vals][:self.n_components]

        # Recast the data along the principal components axes
        project_X = np.dot(feat_vect.T, norm_X.T)

        self.components = feat_vect
        # Return projected data and
        return project_X, feat_vect