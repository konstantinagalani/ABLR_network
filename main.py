import numpy as np
from Optimize_dataset import Pr_dataset
from Optimize_function import Pr_function

def main():
    """
    Main fucntion that tests all the methods depending on the oblective function
    Quadratic function : f = a*x^2 + b*x + 3*c, 
    x in [-5,5], (a,b,c) in [0.1,10], T=30,N=10,C=3,dim_x=1,random_runs=10,add_coef=True
    Branin function
    -5 <= x <= 10, 0 <= y <= 15, for 300 repetitions : T=30,N=10,C=3,dim_x=2,random_runs=10,add_coef=False
    Dataset : ECML dataset - svm and databoost
    T-optional : how many files from each dataset, random_runs = 10
    """
    C = 3
    obj_fun = input("Find the minimum of:")
    print(obj_fun)
    if obj_fun == "Quadratic function":
        T = 30
        N = 10
        dim_x = 3
        random_runs =10
        add_coef = True
        test = Pr_function(np.array([-5,-5,-5]), np.array([5,5,5]),T,N,C,dim_x,random_runs,add_coef)
    elif obj_fun == "Branin function":
        T = 100
        N = 10
        dim_x = 2
        random_runs = 2
        add_coef = False
        test = Pr_function(np.array([-5,0]), np.array([10,15]),T,N,C,dim_x,random_runs,add_coef)
    elif obj_fun == "Dataset":
        T = 35
        random_runs = 10
        test = Pr_dataset(T,random_runs)
    test.run()

main()