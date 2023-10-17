from sklearn import mixture

from E2USD.abstractions import *


class DPGMM(BasicClusteringClass):
    def __init__(self, n_states, alpha=1e3):
        self.alpha = alpha
        if n_states is not None:
            self.n_states = n_states
        else:
            self.n_states = 20

    def fit(self, X):
        dpgmm = mixture.BayesianGaussianMixture(init_params='kmeans',
                                                n_components=self.n_states,
                                                covariance_type="full",
                                                weight_concentration_prior=self.alpha, # alpha
                                                weight_concentration_prior_type='dirichlet_process',
                                                max_iter=1000).fit(X)
        return dpgmm.predict(X)
