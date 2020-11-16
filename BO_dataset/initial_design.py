import numpy as np
from scipy import spatial

class InDesign():

    def __init__(self, x_design):

        self.x_design = x_design


    def initial_design_uniform(self, lower, upper, n_points, rng):
        """
        Nearest neighbour search for uniformly sampled points from the dataset
        lower: np.ndarray (D)
        Lower bounds of the input space
        upper: np.ndarray (D)
        Upper bounds of the input space
        n_points: int
        The number of initial data points
        rng: np.random.RandomState
        Random number generator
        Returns
        -------
        np.ndarray(N,D)
        The initial design data points, that are included in the dataset
        """

        n_dims = lower.shape[0]

        X_correct = self.x_design # get the actual input from the dataset
        X = np.array([rng.uniform(lower, upper, n_dims) for _ in range(n_points)])
        X = np.round(X,4)
        tree = spatial.KDTree(X_correct)
        _,index = tree.query(X)
        points = np.array([X_correct[i] for i in index]).reshape((n_points,n_dims))
        return points
