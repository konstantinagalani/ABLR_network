import numpy as np
import matplotlib.pyplot as plt
import george
import scipy
import robo
from functions.extract_dataset import Get_data
from functions.Function_dataset import Objective_function
from Bo_dataset.initial_design import InDesign
from Bo_dataset.initial_design_random import Rand_Design
from Net import Net
from Bo_dataset.gaussian import GaussianProcess
from plot import Plot_
from Bo_dataset.maximizer import RandomSampling
import torch as t
from george import kernels
from math import log
from scipy.stats import multivariate_normal as mv_norm
from robo.acquisition_functions.ei import EI
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.priors.default_priors import DefaultPrior
from random_S import random_search
from robo.models.gaussian_process import GaussianProcess
from scipy.optimize import differential_evolution, nnls, minimize
from scipy.stats import mode


class Pr_dataset():

    def __init__(self, T=3, random_runs=2, net=2):
        """
           T : number of files for each dataset
           net : int
           if net=1 transfer ABLR is optimized first on T-1 tasks and then on T
           if net=2 transfer ABLR is optimized the sum of log-likelihoods optimized is for all tasks
        """
        self.C_nctx = 0
        self.random_runs = random_runs
        self.T = T
        self.net = net


    def bo(self,fun,seed, model_net,indes,randdes):
        """
           Bayesian Optimization for ABLR
            Parameters:
            fun: Function_dataset object
                mapping of the data
            model: Net Object
            seed : int
            indes : Initial_design Object
        """
        # BO for the network
        lower = np.zeros((self.x.shape[2]))
        upper = np.ones((self.x.shape[2]))
        rng = np.random.RandomState(seed)
        cov_amp = 2
        n_dims = self.x.shape[2]
        initial_ls = np.ones([n_dims])
        exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                              ndim=n_dims)
        kernel = cov_amp * exp_kernel
        prior = DefaultPrior(len(kernel) + 1)
        n_hypers = 3 * len(kernel)

 
        if n_hypers % 2 == 1:
            n_hypers += 1

        model = model_net
        acq = EI(model)
        f = fun
        #max_func = RandomSampling(acq,lower,upper,indes,n_samples=300,rng=rng)
        max_func = RandomSampling(acq,lower,upper,randdes,n_samples=300,rng=rng)
        bo = BayesianOptimization(
            f.call, 
            lower=lower,
            upper=upper,
            acquisition_func=acq, 
            model=model, 
            initial_design=randdes.initial_design_random,
            initial_points=3,
            rng=rng,
            maximize_func=max_func)

        bo.run(num_iterations=50)
        return bo.incumbents_values 


    def Gp(self,seed):
        """
        Bayesian optimization with Gaussian process(mcmc)
        """
        lower = np.zeros((self.x.shape[2]))
        upper = np.ones((self.x.shape[2]))
        inc = np.ones((self.T,50))
        for t in range(self.T):
            rng = np.random.RandomState(seed) # this per task per random run
            cov_amp = 2
            n_dims = self.x.shape[2]
            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                              ndim=n_dims)
            kernel = cov_amp * exp_kernel
            prior = DefaultPrior(len(kernel) + 1)
            n_hypers = 3 * len(kernel)

 
            if n_hypers % 2 == 1:
                n_hypers += 1

            model = GaussianProcess(
            kernel, 
            prior=prior, 
            rng=rng,
            normalize_output=False, 
            normalize_input=False,
            noise=1e-6)
            acq = EI(model)
            # for the initial design, initialize the class with x-task
            f = Objective_function(self.x[t],self.y[t],self.metadata[t])
            indes= InDesign(self.x[t])
            randdes = Rand_Design(self.x[t])
            max_func = RandomSampling(acq,lower,upper,randdes,n_samples=100,rng=rng)

            bo = BayesianOptimization(
            f.call, 
            lower=lower,
            upper=upper,
            acquisition_func=acq, 
            model=model, 
            initial_design=randdes.initial_design_random,
            initial_points=3,
            rng=rng,
            maximize_func=max_func)

            bo.run(num_iterations=50)
            inc[t] = bo.incumbents_values

        return inc




    def ABLR_dataset_optimize(self,dim_x,seed):
        """
        Initialize the network for all the three cases : Simple ABLR, ABLR transfer and ABLR transfer ctx
        Parameters:
        X_init : np.array(T,N,dim_x)
        y_init : np.array(T,N,1)
        seed : int
        Returns dictionary with the incumbents

        """
        rng = np.random.RandomState(seed)
        rng1 = np.random.RandomState(seed)
        inc_simple = np.ones((self.T,50))
        inc_tr = np.ones((self.T,50))
        inc_cont_tr = np.ones((self.T,50))

        #Initialize net for each case
        model_simple = np.ndarray((self.T,),dtype=np.object)
        model_tr = np.ndarray((self.T,),dtype=np.object)
        model_cont_tr = np.ndarray((self.T,),dtype=np.object)
        # Concatenate X with metafeatures
        self.X_all = np.dstack((self.X_init,self.meta_init)).reshape((self.T,self.N,self.C+dim_x))

        if self.net == 2:

            transfer_model = Net(self.C_nctx, self.T, self.N, transfer=True, 
                                  ctx=False,dim_x=dim_x, dataset=True,net=self.net) #one model for the log-likelihood optimization
            transfer_context_model = Net(self.C, self.T, self.N, transfer=True, 
                                       ctx=True,dim_x=dim_x, dataset=True,net=self.net)
            # Optimize only once over all tasks
            transfer_model.x = t.from_numpy(self.X_init).reshape((self.T,self.N,dim_x)).float()
            transfer_model.y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
            # for train in Bayesian optimization
            transfer_model.X_init = self.X_init
            transfer_model.Y_init = self.Y_init
            transfer_model.optimize()
            print("ABLR transfer LBFGS")
            # Optimize only once over all tasks-ctx
            transfer_context_model.x = t.from_numpy(self.X_all).reshape((self.T,self.N,self.C + dim_x)).float()
            transfer_context_model.y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
            # for train in Bayesian optimization
            transfer_context_model.X_init = self.X_all
            transfer_context_model.Y_init = self.Y_init
            transfer_context_model.optimize()
            print("ABLR context transfer LBFGS")

        for i in range(self.T):

            f = Objective_function(self.x[i],self.y[i],self.metadata[i])
            # Initialize model for each task
            model_simple[i] = Net(self.C_nctx,self.T, self.N, transfer=False, ctx=False,dim_x=dim_x, dataset=True, f=f) #ABLR simple

            if self.net == 1:

                model_tr[i] = Net(self.C_nctx, self.T, self.N, transfer=True, 
                                  ctx=False,dim_x=dim_x, dataset=True, f=f,net=self.net) #ABLR transfer
                model_cont_tr[i] = Net(self.C, self.T, self.N, transfer=True, 
                                       ctx=True,dim_x=dim_x, dataset=True,f=f,net=self.net) #ABLR transfer context, pass function to get the contextual information
                #Transfer phase 1: optimize with X_init, Y_init
                model_tr[i].x = t.from_numpy(self.X_init).reshape((self.T,self.N,dim_x)).float()
                model_tr[i].y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                model_tr[i].i = i
                model_tr[i].optimize()
                #Transfer phase 2 : optimize with new x and y for task i
                model_tr[i].x = t.from_numpy(self.X_init[i]).reshape((self.N,dim_x)).float()
                model_tr[i].y = t.from_numpy(self.Y_init[i]).reshape((self.N,1)).float()
                model_tr[i].do_optimize = True
                model_tr[i].optimize()
                print("ABLR transfer LBFGS",i)

                #Transfer context phase 1: optimize with X_init, Y_init but concatenated with metadata
                model_cont_tr[i].x = t.from_numpy(self.X_all).reshape((self.T,self.N,self.C+dim_x)).float()
                model_cont_tr[i].y = t.from_numpy(self.Y_init).reshape((self.T,self.N,1)).float()
                model_cont_tr[i].i = i
                model_cont_tr[i].optimize()
                #Transfer context phase 2 : optimize with new x and y for task i
                model_cont_tr[i].x = t.from_numpy(self.X_all[i]).reshape((self.N,self.C+dim_x)).float()
                model_cont_tr[i].y = t.from_numpy(self.Y_init[i]).reshape((self.N,1)).float()
                model_cont_tr[i].do_optimize = True
                model_cont_tr[i].optimize()
                print("ABLR transfer context LBFGS",i)

            #Initialize initial design
            indes= InDesign(self.x[i])
            randdes = Rand_Design(self.x[i])
            if self.net == 2:
                model_tr[i] = transfer_model
                model_tr[i].i = i
                model_tr[i].f = f
                model_cont_tr[i] = transfer_context_model
                model_cont_tr[i].i = i
                model_cont_tr[i].f = f
            

            #BO ABLR trasfer ctx
            print("BO ABLR transfer ctx")
            inc_cont_tr[i] = self.bo(f,seed,model_cont_tr[i],indes, randdes)
            #BO ABLR transfer
            print("BO ABLR transfer")
            inc_tr[i] = self.bo(f,seed,model_tr[i],indes, randdes)
            #BO ABLR simple
            print("BO ABLR simple")
            inc_simple[i] = self.bo(f,seed,model_simple[i],indes, randdes)

        total_inc = dict()
        total_inc["simple"] = inc_simple
        total_inc["transfer"]  = inc_tr
        total_inc["transfer context"] = inc_cont_tr

        return total_inc


    def run(self):

        # Get the data
        data = Get_data()
        self.y,self.x, self.metadata = data.read_file()# y: (number of tasks,N), x:(number of tasks,100,dim_x), metadata: (number of tasks,100, C)
        self.x = self.x.astype(np.float32)
        self.y = 1 - self.y
        #self.T = self.x.shape[0] # how many files in the dataset
        #self.N = self.x.shape[1] # how many lines in the file
        self.N = 30
        dim_x = self.x.shape[2]
        self.C = self.metadata.shape[2]
        self.X_init = np.ones((self.T,self.N,dim_x))
        self.Y_init = np.ones((self.T,self.N,1))
        self.meta_init = np.ones((self.T,self.N,self.C))
        #Initialize incumbents
        inc_rand = np.ones((self.T,50))
        inc_all_rand = np.ones((self.random_runs,self.T,50))
        inc_all_gp = np.ones((self.random_runs,self.T,50))
        inc_all_ABLR_simple = np.ones((self.random_runs,self.T,50))
        inc_all_ABLR_tr = np.ones((self.random_runs,self.T,50))
        inc_all_ABLR_cont_tr = np.ones((self.random_runs,self.T,50))


        for r in range(self.random_runs):
            for t in range(self.T):
                seed = t + r
                rng = np.random.RandomState(seed)
                results = random_search(dim_x=dim_x, N=self.N, X_init=self.x[t],Y_init=self.y[t],num_iterations=50, rng=rng, dataset=True)
                inc_rand[t] = results["incumbent_values"]

                #Initial points for transfer learning
                line = [i for i in range(self.x[t].shape[0])]
                idx = rng.choice(line, size=self.N,replace=False) # choose random x from the file
                self.X_init[t] = np.array([self.x[t][idx[n]] for n in range(self.N)]).reshape((self.N,dim_x))
                self.Y_init[t] = np.array([self.y[t][idx[n]] for n in range(self.N)]).reshape((self.N,1))
                self.meta_init[t] = np.array([self.metadata[t][idx[n]] for n in range(self.N)]).reshape((self.N,self.C))
            # Random search
            print("Random",r)
            inc_all_rand[r] = inc_rand
            # GP plain
            print("Gp",r)
            inc_all_gp[r] = self.Gp(r)
            #ABLR
            total_inc = self.ABLR_dataset_optimize(dim_x,seed)
            inc_all_ABLR_tr[r] = total_inc['transfer']
            inc_all_ABLR_cont_tr[r] = total_inc['transfer context']
            inc_all_ABLR_simple[r] = total_inc['simple']
        # take the mean for each task for all random runs
        #pl = Plot_(self.T, np.mean(inc_all_ABLR_simple, axis=0),np.mean(inc_all_ABLR_tr,axis=0),
                 # np.mean(inc_all_ABLR_cont_tr,axis=0),np.mean(inc_all_rand,axis=0), np.mean(inc_all_gp,axis=0))
        #y = self.y[:self.T] # change this when i run it for all files
        #ymin = np.amin(y, axis=1).reshape((self.T,1))
        #ymax = np.amax(y, axis=1).reshape((self.T,1))
        #pl.plot_adtm(ymin,ymax)