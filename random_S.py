import os
import time
import numpy as np


def random_search(dim_x,N,objective_function=None, lower=None, upper=None, X_init=[], Y_init=[],
                  num_iterations=30, rng=None, dataset=False):
    """
    Random Search [1] that simply evaluates random points. We do not have
    any priors thus we sample points uniformly at random.
    [1] J. Bergstra and Y. Bengio.
        Random search for hyper-parameter optimization.
        JMLR, 2012.
    Parameters
    ----------
    objective_function: function
        Objective function that will be optimized
    lower: np.array(D,)
        Lower bound of the input space
    upper: np.array(D,)
        Upper bound of the input space
    X_init: np.array (T,N, D)
            Points from dataset
    Y_init: np.array (T, N,1)
            Function values from dataset
    num_iterations: int
        Number of iterations
    rng: numpy.random.RandomState
        Random number generator
    dataset : boolean
        True if working in dataset
    Returns
    -------
    dict with all results
    """

    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState()

    time_func_evals = []
    time_overhead = []
    incumbents = []
    incumbents_values = []
    runtime = []

    X = []
    y = []

    if dataset: 
        line = [i for i in range(X_init.shape[0])]
        idx = rng.choice(line, size=num_iterations,replace=False)

    for it in range(num_iterations):
        # Choose next point to evaluate
        if not dataset:
            x = rng.uniform(lower,upper,size=(1,dim_x))
            new_y = objective_function(x)
        else:
            # choose a point from the available ones from the file
            x = X_init[idx[it]]
            new_y = Y_init[idx[it]]


        # Update the data
        X.append(x)
        y.append(new_y)

        # The incumbent is just the best observation we have seen so far
        best_idx = np.argmin(y)
        incumbent = X[best_idx]
        incumbent_value = y[best_idx]

        incumbents.append(incumbent)
        incumbents_values.append(incumbent_value)


    results = dict()
    results["x_opt"] = incumbent
    results["f_opt"] = incumbent_value
    results["incumbents"] = incumbents
    results["incumbent_values"] = incumbents_values
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_evals
    results["X"] = X
    results["y"] = y

    return results