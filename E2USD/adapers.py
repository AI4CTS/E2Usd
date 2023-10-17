import numpy as np
import sys
import os
from E2USD.utils import all_normalize
sys.path.append(os.path.dirname(__file__))
import encoders
from E2USD.abstractions import *

class E2USD_Adaper(BasicEncoderClass):
    def _set_parmas(self, params):
        self.hyperparameters = params
        if 'compared_length' in self.hyperparameters:
            del self.hyperparameters['compared_length']
        self.encoder = encoders.DDEM(**self.hyperparameters)

    def fit(self, X):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        self.encoder.fit(X, save_memory=True, verbose=False)

    def encode(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:, :], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

    def encode_one(self, X, win_size, step):
        _, dim = X.shape
        X = np.transpose(np.array(X[:,:], dtype=float)).reshape(1, dim, -1)
        X = all_normalize(X)
        embeddings = self.encoder.encode_window(X, win_size=win_size, step=step)
        return embeddings

