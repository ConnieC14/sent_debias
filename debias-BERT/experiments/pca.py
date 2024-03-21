import numpy as np

class PCA_():
    def __init__(
        self,
        n_components=None
    ):
        self.n_components = n_components

    def fit(self, X, svd=True):
        # Center data
        mean = np.mean(X, axis=0)
        # std = np.std(X)
        norm_X = (X - mean)
        
        if svd:
            # Calculate SVD based on sklearn's pca
            # Skips covariance matrix and eigenvector steps
            # A = US*(vh)
            u, s, Vt = np.linalg.svd(norm_X, full_matrices=False)
            
            # apply this for sign consistency
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u = u * signs
            Vt *= signs[:, np.newaxis]

            Vt = Vt[:self.n_components].T
            # Do not whiten
            project_X = u[:, :self.n_components]
            # proj_X = X * V = U * S * Vt * V = U * S
            project_X = project_X * s[: self.n_components]
            self.projected_data = project_X

            components_ = Vt.T
            self.components_ = components_
        else:
            # Compute covariance matrix to identify correlations
            cov_X = np.dot(norm_X.T, norm_X) / (X.shape[0] - 1)
            print("covariance: ", cov_X)

            # Compute the eigenvectors and eigenvalues of the covariance matrix to identify the principal components
            eig_values, eig_vectors = np.linalg.eig(cov_X)

            # Create a feature vector to decide which principal components to keep
            idx_vals = np.argsort(eig_values)[::-1]
            feat_vect = eig_vectors[idx_vals][:, :self.n_components]

            # Recast the data along the principal components axes
            project_X = np.dot(feat_vect.T, norm_X.T)
            self.projected_data = project_X

            components_ = feat_vect
            self.components_ = components_
        
        # Return projected data and
        return project_X, components_

# from sklearn.decomposition import PCA
# from pca import PCA_
# import unittest
# import numpy as np
# import numpy as np

# num_components = 1

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# print(X.shape)

# pca = PCA(n_components=num_components, svd_solver="auto")
# y = pca.fit(X)

# pca_ = PCA_(n_components=num_components)
# projected_data, feat_vect = pca_.fit(X)

# print(f"\nshapes: {(y.components_.shape, feat_vect.shape)}\n")
