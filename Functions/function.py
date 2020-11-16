import numpy as np

class Quadratic_function():
    """
       Quadratic function defined for experiments in "Scalable Hyperparameter Transfer Learning", Perone er al.
    """
    def __init__(self, coef):

        self.a = coef[0]
        self.b = coef[1]
        self.c = coef[2]
    
    def call(self, x):
    
        if x.shape[0]==1 or len(x.shape)==1: # if we have one data point
            y = 1/2*self.a*np.linalg.norm(x)**2 + self.b*np.sum(x)+ 3*self.c
        else:
            y = 1/2*self.a*np.linalg.norm(x, axis=1)**2 + self.b*np.sum(x, axis=1) + 3*self.c # noisy y data (tensor)
        y = y.astype(np.float32)
        return y
