'''
Created by Chengyu on 2022/2/26.
'''

import numpy as np
from TSpy.label import reorder_label
from TSpy.utils import calculate_scalar_velocity_list

class Time2State:
    def __init__(self, win_size, step, encoder, clustering_component, verbose=False):
        """
        Initialize Time2State_backup.

        Parameters
        ----------
        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        encoder_class : object.
            The instance of encoder.

        clustering_class: object.
            The instance of clustering component.
        """

        # The window size must be an even number.
        if win_size%2 != 0:
            raise ValueError('Window size must be even.')

        self.__win_size = win_size
        self.__step = step
        self.__offset = int(win_size/2)
        self.__encoder = encoder
        self.__clustering_component = clustering_component

    def fit(self, X, win_size, step):
        """
        Fit time2state.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        
        self.__length = X.shape[0]
        self.fit_encoder(X)
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def predict(self, X, win_size, step):
        """
        Find state sequence for X.

        Parameters
        ----------
        X : {ndarray} of shape (n_samples, n_features)

        win_size : even integer.
            The size of sliding window.

        step : integer.
            The step size of sliding window.

        Returns
        -------
        self : object
            Fitted time2state.
        """
        self.__length = X.shape[0]
        self.__step = step
        self.__encode(X, win_size, step)
        self.__cluster()
        self.__assign_label()
        return self

    def set_step(self, step):
        self.__step = step

    def set_clustering_component(self, clustering_obj):
        self.__clustering_component = clustering_obj
        return self

    def fit_encoder(self, X):
        self.__encoder.fit(X)
        return self

    def predict_without_encode(self, X, win_size, step):
        self.__cluster()
        self.__assign_label()
        return self

    def __encode(self, X, win_size, step):
        self.__embeddings = self.__encoder.encode(X, win_size, step)

    def __cluster(self):
        self.__embedding_label = reorder_label(self.__clustering_component.fit(self.__embeddings))

    def __assign_label(self):
        hight = len(set(self.__embedding_label))
        # weight_vector = np.concatenate([np.linspace(0,1,self.__offset),np.linspace(1,0,self.__offset)])
        weight_vector = np.ones(shape=(2*self.__offset)).flatten()
        self.__state_seq = self.__embedding_label
        vote_matrix = np.zeros((self.__length,hight))
        i = 0
        for l in self.__embedding_label:
            vote_matrix[i:i+self.__win_size,l]+= weight_vector
            i+=self.__step
        self.__state_seq = np.array([np.argmax(row) for row in vote_matrix])

    def save_encoder(self):
        pass

    def load_encoder(self):
        pass

    def save_result(self, path):
        pass

    def load_result(self, path):
        pass

    def plot(self, path):
        pass

    @property
    def embeddings(self):
        return self.__embeddings

    @property
    def state_seq(self):
        return self.__state_seq
    
    @property
    def embedding_label(self):
        return self.__embedding_label

    @property
    def velocity(self):
        return self.__velocity

    @property
    def change_points(self):
        return self.__change_points