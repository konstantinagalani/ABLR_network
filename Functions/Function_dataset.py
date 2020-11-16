import numpy as np

class Objective_function():

    def __init__(self, X ,Y,md):
        """
        Create a dictionary that contains data from a specific task, where X is the key and Y is the value
        input:
        X : np.array(number of lines, C), dtype=float
        Y : np.array(number of lines,1), dtype=float
        """ 
        self.mapd = dict()
        self.metafeatures = dict()

        for i in range(X.shape[0]):
            key = str(X[i])
            self.mapd[key] = Y[i]
            self.metafeatures[key] = md[i]
    
    
    def call(self,x_new):
    	"""
    	x_new : np.array(1,C), dtype=float
    	returns
    	y : np.float
    	"""
    	x = str(x_new)
    	y = self.mapd[x]
    	y = y.astype(np.float32)
    	return y
