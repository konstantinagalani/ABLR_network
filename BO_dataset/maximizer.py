import numpy as np



class RandomSampling():

    def __init__(self, objective_function, lower, upper, init_d,n_samples=100, rng=None):
        """
        Samples candidates uniformly at random and returns the point with the highest objective value.
        Parameters
        ----------
        objective_function: acquisition function
            The acquisition function which will be maximized
        lower: np.ndarray (D)
            Lower bounds of the input space
        upper: np.ndarray (D)
            Upper bounds of the input space
        init_d : initial design object
            Object used to sample points from the dataset
        n_samples: int
            Number of candidates that are samples
        """
        self.n_samples = n_samples
        self.init_d = init_d
        self.lower = lower
        self.upper = upper
        self.objective_func = objective_function
        self.rng = rng


    def maximize(self):
        """
        Maximizes the given acquisition function.
        Returns
        -------
        np.ndarray(N,D)
            Point with highest acquisition value.
        """

        # Sample random points uniformly over the whole space

        X = self.init_d.initial_design_random(self.lower, self.upper,self.n_samples, self.rng)
        y = np.array([self.objective_func(X[i].reshape((1,X[i].shape[0]))) for i in range(self.n_samples)])

        x_star = X[y.argmax()]

        return x_star