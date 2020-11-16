import numpy as np
import matplotlib.pyplot as plt
import george
import scipy
import robo
from functions.function import Quadratic_function
from functions.branin import Branin_function
from Net import Net
from plot import Plot_
import torch as t
from george import kernels
from math import log
from scipy.stats import multivariate_normal as mv_norm
from robo.maximizers.scipy_optimizer import SciPyOptimizer
from robo.initial_design.init_latin_hypercube_sampling import init_latin_hypercube_sampling
from robo.acquisition_functions.ei import EI
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.priors.default_priors import DefaultPrior
from robo.fmin import bayesian_optimization
from random_S import random_search
from scipy.optimize import differential_evolution, nnls, minimize
from scipy.stats import mode

class Pr_function():
# Implemented objective functions : branin, quadratic function with coefficients

    def __init__(self, lower_x, upper_x,T=7, N=10, C=3, dim_x=1, random_runs=4, add_coef=False,net=2):
        """
        Initialize class for functions
        Parameters:
        lower_x : np.array(dim_x,)
        upper_x : np.array(dim_x,)
        T : int
        number of tasks (in branin there is only one task so T counts as repetition)
        C, C_nctx : int
        shape of contextual information
        add_coef : bool
        True if the function contains contextual information that depends on the task
        net : int
        if net=1 transfer ABLR is optimized first on T-1 tasks and then on T
        if net=2 transfer ABLR is optimized the sum of log-likelihoods optimized is for all tasks
        """
        self.lower_x = lower_x
        self.upper_x = upper_x
        self.T = T
        self.N = N
        self.C = C
        self.dim_x = dim_x
        self.C_nctx = 0
        self.random_runs = random_runs
        self.add_coef = add_coef
        self.net = net

    def bo(self,model, seed):
        """ Bayesian Optimization for ABLR
            Parameters:
            model: Net Object
            seed : int
        """
        rng = np.random.RandomState(seed)
        model1 = model
        acq = EI(model1)
        max_func = SciPyOptimizer(acq, self.lower_x, self.upper_x)

        if model1.transfer : init = 1 # 1 initial point for transfer case
        else: init = 5

        bo = BayesianOptimization(
            self.f.call, 
            lower=self.lower_x, 
            upper=self.upper_x, 
            acquisition_func=acq, 
            model=model1,
            maximize_func=max_func, 
            initial_points=init,
            initial_design=init_latin_hypercube_sampling,
            rng=rng)

        bo.run(num_iterations=50)

        return bo.incumbents_values
    

    def gp_plain(self,seed,num_iterations=50):
        """Bayesian optimization with Gaussian Process(mcmc) as model and no contextual information

           Returns np.array(1,50) mean of incumbents
        """
        incumbent_value = np.ones((self.T,num_iterations))
        for i in range(self.T):
            rng1 = np.random.RandomState(seed)
            if self.add_coef : self.f = Quadratic_function(self.coef[i])
            result = bayesian_optimization(self.f.call, self.lower_x, self.upper_x, num_iterations=num_iterations,
                            model_type="gp", acquisition_func="ei", n_init=1,rng=rng1)
            #Branin needs init=1
            incumbent_value[i] = result["incumbent_values"]
        return np.mean(incumbent_value, axis=0)


    def ABLR_fun_optimize(self,seed):
        """
        Initialize and optimize the network for all the three cases : Simple ABLR, ABLR transfer and ABLR transfer ctx
        Parameters:
        X_init : np.array(T,N,dim_x) without contextual infromation!
        y_init : np.array(T,N,1)
        seed : int

        Returns dictionary with the incumbents
        """
        rng1 = np.random.RandomState(seed)
        rng = np.random.RandomState(seed)
        inc_simple = np.ones((self.T,50))
        model_simple = np.ndarray((self.T,),dtype=np.object)

        if self.add_coef: # Initialize for ABLR transfer
            inc_tr = np.ones((self.T,50))
            inc_cont_tr = np.ones((self.T,50))
            model_tr = np.ndarray((self.T,),dtype=np.object)
            model_cont_tr = np.ndarray((self.T,),dtype=np.object)

            if self.net == 2:
                #for the network self.optimize is True only in predict
                transfer_model = Net(self.C_nctx, self.T, self.N, transfer=True, 
                                  ctx=False,dim_x=self.dim_x, dataset=False, net=self.net) #one model for the log-likelihood optimization
                transfer_context_model = Net(self.C, self.T, self.N, transfer=True, 
                                       ctx=True,dim_x=self.dim_x, dataset=False,net=self.net)
                # Optimize only once over all tasks
                transfer_model.coef = self.coef
                transfer_model.x = t.from_numpy(self.X_init).reshape((self.T,self.N,self.dim_x)).float()
                transfer_model.y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                transfer_model.optimize()
                print("ABLR transfer LBFGS")
                # Optimize only once over all tasks-ctx
                transfer_context_model.coef = self.coef
                transfer_context_model.x = t.from_numpy(self.X_all).reshape((self.T,self.N,self.dim_x+self.C)).float()
                transfer_context_model.y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                transfer_context_model.optimize()
                print("ABLR context transfer LBFGS")

        for i in range(self.T):
            # Initialize model for each task
            if self.add_coef : self.f = Quadratic_function(self.coef[i])
            model_simple[i] = Net(self.C_nctx,self.T, self.N, transfer=False, ctx=False,dim_x=self.dim_x, f=self.f) #ABLR simple

            if self.add_coef:
                if self.net == 1:
                    model_tr[i] = Net(self.C_nctx, self.T, self.N, transfer=True, ctx=False,dim_x=self.dim_x) #ABLR transfer
                    model_cont_tr[i] = Net(self.C, self.T, self.N, transfer=True, ctx=True,dim_x=self.dim_x) #ABLR transfer context


                    #Transfer phase 1: optimize with X_init, Y_init
                    model_tr[i].coef = self.coef
                    model_tr[i].x = t.from_numpy(self.X_init).reshape((self.T,self.N,self.dim_x)).float()
                    model_tr[i].y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                    model_tr[i].i = i
                    model_tr[i].optimize()
                    #Transfer phase 2 : optimize with new x and y for task i
                    model_tr[i].x = t.from_numpy(self.X_init[i]).reshape((self.N,self.dim_x)).float()
                    model_tr[i].y = t.from_numpy(self.Y_init[i]).reshape((self.N,1)).float()
                    model_tr[i].do_optimize = True
                    model_tr[i].optimize()
                    print("ABLR transfer LBFGS",i)

                    #Transfer context phase 1: optimize with X_init, Y_init but concatenated with metadata
                    model_cont_tr[i].coef = self.coef
                    model_cont_tr[i].x = t.from_numpy(self.X_all).reshape((self.T,self.N,self.C+self.dim_x)).float()
                    model_cont_tr[i].y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                    model_cont_tr[i].i = i
                    model_cont_tr[i].optimize()
                    #Transfer context phase 2 : optimize with new x and y for task i
                    model_cont_tr[i].x = t.from_numpy(self.X_all[i]).reshape((self.N,self.C+self.dim_x)).float()
                    model_cont_tr[i].y = t.from_numpy(self.Y_init[i]).reshape((self.N,1)).float()
                    model_cont_tr[i].do_optimize = True
                    model_cont_tr[i].optimize()
                    print("ABLR transfer context LBFGS",i)

                if self.net == 2:
                    model_tr[i] = transfer_model
                    model_tr[i].i = i
                    model_cont_tr[i] = transfer_context_model
                    model_cont_tr[i].i = i

                # for train in Bayesian optimization-transfer
                model_tr[i].X_init = self.X_init
                model_tr[i].Y_init = self.Y_init
                # for train in Bayesian optimization-transfer ctx
                model_cont_tr[i].X_init = self.X_all
                model_cont_tr[i].Y_init = self.Y_init
                #BO ABLR transfer ctx
                print("BO ABLR transfer ctx")
                inc_cont_tr[i] = self.bo(model_cont_tr[i], seed)
                #BO ABLR transfer
                print("BO ABLR transfer")
                inc_tr[i] = self.bo(model_tr[i], seed)

            #BO ABLR simple
            print("BO ABLR simple")
            inc_simple[i] = self.bo(model_simple[i], seed)
        total_inc = dict()
        total_inc["simple"] = np.mean(inc_simple, axis=0)
        if self.add_coef:
            total_inc["transfer"]  = np.mean(inc_tr, axis=0)
            total_inc["transfer context"] = np.mean(inc_cont_tr, axis=0)
        return total_inc
        



    def run(self):
    
        if self.add_coef:
            self.coef = np.ones((self.T,self.C))
        self.X_init = np.ones((self.T,self.N,self.dim_x))
        self.X_all = np.ones((self.T,self.N,self.dim_x+self.C)) # initial points concatinated with coefficients
        self.Y_init = np.ones((self.T,self.N,1))
        # Incumbents for all the cases
        inc_all_rand = np.ones((self.random_runs,50))
        inc_all_gp = np.ones((self.random_runs,50))
        inc_all_ABLR_simple = np.ones((self.random_runs,50))
        inc_all_ABLR_tr = np.ones((self.random_runs,50))
        inc_all_ABLR_cont_tr = np.ones((self.random_runs,50))

        # 10 random runs for each method
        for j in range(self.random_runs):
            # for all the tasks get the coeficients and the N*T observations for the transfer cases
            incumbents_rand = np.ones((self.T,50))# for random search
            for i in range(self.T):
                seed = self.T  +5*i + 3*j
                rng = np.random.RandomState(seed)
                if self.add_coef: 
                    # get coefficients for the tasks
                    lower_c = np.array([0.1, 0.1, 0.1])
                    upper_c = np.array([10, 10, 10])
                    self.coef[i] = (rng.uniform(lower_c, upper_c)).reshape((1,self.C))
                    self.f = Quadratic_function(self.coef[i])
                    # Initial points for transfer learning
                    x = rng.uniform(self.lower_x, self.upper_x, size=(self.N,self.dim_x))
                    x_new = ((x - np.mean(x, axis=0))/np.std(x, axis=0)).reshape(self.N,self.dim_x) # normalize input
                    self.Y_init[i] = t.from_numpy(self.f.call(x_new)).reshape((self.N,1)).float()
                    self.X_init[i] = x_new

                    # concatenate x with coefficients
                    ctx_new = np.tile(self.coef[i],(self.N,1))
                    self.X_all[i] = np.concatenate((self.X_init[i], ctx_new), axis=1).reshape((self.N,self.C + self.dim_x))
                else: 
                    self.f = Branin_function()
                #random search
                rng = np.random.RandomState(self.T  +5*i + 3*j)
                results = random_search(self.dim_x, self.N,self.f.call,self.lower_x, self.upper_x,num_iterations=50, rng=rng,dataset=False)
                incumbents_rand[i] = results["incumbent_values"]
        

            # Random search
            print("Random search", j)
            inc_all_rand[j] = np.mean(incumbents_rand, axis=0)
            #GP Plain
            print("GP plain", j)
            inc_all_gp[j] = self.gp_plain(seed)
            # Train ABLR
            print("ABLR", j)
            total_inc = self.ABLR_fun_optimize(seed)
            if self.add_coef:
                inc_all_ABLR_tr[j] = total_inc["transfer"]
                inc_all_ABLR_cont_tr[j] = total_inc["transfer context"]
                transfer = True
            else:
               transfer = False
            inc_all_ABLR_simple[j] = total_inc["simple"]

        #p = Plot_(self.T, inc_all_ABLR_simple,inc_all_ABLR_tr, inc_all_ABLR_cont_tr, 
                  #inc_all_rand, inc_all_gp,transfer=transfer)
        #p.plot_simple()
