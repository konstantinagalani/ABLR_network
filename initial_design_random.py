import numpy as np

class Rand_Design():

    def __init__(self, x_design):

        self.x_design = x_design
        # contains the x of the specific task
    
    def initial_design_random(self, lower, upper, n_points, rng):
        """
        Chooses random points from the existing dataset
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


        line = [i for i in range(self.x_design.shape[0])]
        idx = rng.choice(line, size=n_points,replace=False) # choose random x from the file
        X_init = np.array([self.x_design[idx[n]] for n in range(n_points)]).reshape((n_points,self.x_design.shape[1]))
        return X_init