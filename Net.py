import numpy as np
import matplotlib.pyplot as plt
import george
import scipy
import robo
import torch.nn as nn
import torch as t
import torch.distributions as d
from george import kernels
from scipy.stats import multivariate_normal as mv_norm
from scipy.optimize import differential_evolution, nnls, minimize
from scipy.stats import mode
from torch.autograd import Variable
from torch.autograd import grad
from functions.function import Quadratic_function
 
D = 50
global c


class Net(t.nn.Module):
    def __init__(self, C, T, N, transfer, ctx, dim_x, dataset=False,f=None,net=2):
        """ 
        c : int
            Size of metadata array for one task
        T : int
            Number of Tasks
        N : int
            Number of data points
        transfer : bool
            True if the network will have a common basis feature matrix across taks
        ctx : bool
            True if the contextual information is added to the network
        dataset : bool
            True if working with the dataset
        f : Objective_function object
        net : int
            net=1 transfer ABLR--> is optimized first on T-1 tasks and then on task T
            net=2 transfer ABLR--> optimize the sum of log-likelihoods for all tasks
        self.dim_x : int
            X.shape[1]
        self.coef : np.array((T,c))
            Metadata array corresponding to each task
        """
        self.c_dim = C
        self.T = T
        self.N = N
        self.dim_x = dim_x
        self.transfer = transfer
        self.ctx = ctx
        self.dataset = dataset
        self.f = f
        self.net = net

        super(Net, self).__init__()
        self.hidden2 = nn.Linear(D, D)
        self.hidden3 = nn.Linear(D, D)

        if self.transfer: # learn common feature basis across tasks, where each task is assigned to a separeate marginal log likelihood
            self.hidden1 = nn.Linear(self.dim_x + self.c_dim, D) # contextual information
            #initialization of alpha and beta
            self.alpha_t = [t.tensor(0.0, requires_grad=True) for i in range(self.T)] # params for each log likelihood
            self.beta_t = [t.tensor(0.0, requires_grad=True) for i in range(self.T)]
            # for the task we are actually optimizing at the time(used in net=1)
            self.alpha = t.tensor(0.0, requires_grad=True)
            self.beta = t.tensor(0.0, requires_grad=True)
        else:
            self.hidden1 = nn.Linear(self.dim_x, D) # simple ABLR
            #initialization of alpha and beta
            self.alpha = t.tensor(0.0, requires_grad=True)
            self.beta = t.tensor(0.0, requires_grad=True)
        # initialization of the weights
        t.nn.init.xavier_normal_(self.hidden1.weight)
        t.nn.init.xavier_normal_(self.hidden2.weight)
        t.nn.init.xavier_normal_(self.hidden3.weight)
        
        self.coef = []
        self.x_train = np.array((1,)) # training points from BO, used when net=2
        self.do_optimize = False # the network is in the first phase (first pass of optimizing)
        self.first = True # first time entering for loop- for the 1st phase of transfer training

        #initialize arrays for the optimization of sum log-likelihood
        self.Phi_t = [t.tensor(0.0) for i in range(self.T)]
        self.K_t = [t.tensor(0.0) for i in range(self.T)]
        self.L_t = [t.tensor(0.0) for i in range(self.T)]
        self.L_inv_t = [t.tensor(0.0) for i in range(self.T)]
        self.c_t = [t.tensor(0.0) for i in range(self.T)]
        self.norm_c_t = [t.tensor(0.0) for i in range(self.T)]
        self.norm_y_t = [t.tensor(0.0) for i in range(self.T)]


        
    def network(self,x):
        # 3- layer network

        x = self.hidden1(x)
        x = t.tanh(x)
        x = self.hidden2(x)
        x = t.tanh(x)
        x= self.hidden3(x)
        self.Phi = t.tanh(x)


        
    def loss(self, hp):
        """
        Negative log marginal likelihood of multi-task ABLR
        hp :np.array
            Contains the weights of the network, alpha and beta
        self.x : np.array((T,N,self.dim_x +c)) 
        self.y : np.array((T,N))
        self.i : which task I am optimizing when the function is called
        """
        c = self.c_dim
        # I have to separate the variables because LBFGS needs a flattened array

        # change the weights with the next proposed weights from LBFGS
        w0 = t.from_numpy(hp[0:(c+self.dim_x)*D].reshape((D,c+self.dim_x)))
        w1 = t.from_numpy(hp[(c+self.dim_x)*D:(c+self.dim_x)*D + D**2].reshape((D,D)))
        w2 = t.from_numpy(hp[(c+self.dim_x)*D + D**2:(c+self.dim_x)*D + 2*D**2].reshape((D,D)))
        self.hidden1.weight.data = w0.float() # change the weights, convert to float32 because the lbfgs gives double
        self.hidden2.weight.data = w1.float()
        self.hidden3.weight.data = w2.float()

        if self.transfer==True and self.do_optimize==False:
            # Net = 2 (One network for all tasks, optimize sum of log likelihoods)
            # Net = 1 (First phase of training with T-1 tasks)
            self.first = True
            self.likelihood = None

            if self.net ==1: x_range = [x for x in range(self.T) if x!=self.i]
            else: x_range = [x for x in range(self.T)]

            for i in x_range:
                if len(self.x_train.shape) == 1:
                    # Warm starting the network(transfer)
                    x_new = self.x[i]
                    y = self.y[i]
                else:
                    if i == self.i: # optimizing this task in BO
                        x_new = self.x_train
                        y = self.y_train
                    else:
                        x_new = self.x[i]
                        y = self.y[i]

                self.network(x_new) 
                self.alpha_t[i].data = t.from_numpy(hp[2*D**2+(c + self.dim_x)*D+2*i].reshape((1,))).float()
                self.beta_t[i].data = t.from_numpy(hp[2*D**2+(c + self.dim_x)*D+2*i+1].reshape((1,))).float()

                # Loss function calculations
                self.Phi_t[i] = self.Phi
                assert(t.t(self.Phi).shape == (D,x_new.shape[0]))
                A = self.alpha_t[i]/self.beta_t[i]*t.matmul(t.t(self.Phi_t[i]), self.Phi_t[i])
                Id = t.eye(D)
                self.K_t[i] = (t.add(A, Id)).double()
                assert(self.K_t[i].shape == (D,D))
                self.L_t[i] = (t.cholesky(self.K_t[i], upper=False)).float() # cholesky factor of K, 50x50
                self.L_inv_t[i] =  t.inverse(self.L_t[i])
                B = t.matmul(t.t(self.Phi_t[i]), y)
                self.c_t[i] = t.matmul(self.L_inv_t[i],B) 
                self.c_t[i] = self.c_t[i].view((D,1))
                assert(self.c_t[i].shape == (D,1))
                self.norm_y_t[i] = t.norm(y,2,0)
                self.norm_c_t[i] = t.norm(self.c_t[i],2,0)
    
                L1 = (self.N/2*t.log(self.alpha_t[i])).clone().detach().requires_grad_(True)
                L2 = -self.alpha_t[i]/2*(t.pow(self.norm_y_t[i],2).add(-self.alpha_t[i]/self.beta_t[i]*t.pow(self.norm_c_t[i],2)))
                L3 = -1*t.trace(t.log(self.L_t[i]))
                sum_L = t.add(L2,L3)

                if self.first: 
                    self.likelihood = t.add(L1,sum_L)
                    self.first = False
                else: 
                    self.likelihood= t.add(self.likelihood,t.add(L1,sum_L))
            self .likelihood = -1*self.likelihood
        else:
            # Net = 1( Transfer case, second pass of optimizing the ith task)
            # ABLR Simple

            x_new = self.x
            self.network(x_new)
            # alpha and beta made tensors so I can get the gradient
            self.alpha.data = t.from_numpy(hp[2*D**2+(c + self.dim_x)*D].reshape((1,))).float()
            self.beta.data = t.from_numpy(hp[2*D**2+(c + self.dim_x)*D+1].reshape((1,))).float()

            # Loss function calculations
            assert(t.t(self.Phi).shape == (D,x_new.shape[0]))
            A = self.alpha/self.beta*t.matmul(t.t(self.Phi), self.Phi)
            Id = t.eye(D) + 1e-10
            self.K = t.add(A, Id).double()
            assert(self.K.shape == (D,D))
            self.L = (t.cholesky(self.K, upper=False)).float() # cholesky factor of K, 50x50
            self.L_inv =  t.inverse(self.L)
            B = t.matmul(t.t(self.Phi), self.y)
            self.c = t.matmul(self.L_inv,B) 
            self.c = self.c.view((D,1))
            assert(self.c.shape == (D,1))
            self.norm_y = t.norm(self.y,2,0)
            self.norm_c = t.norm(self.c,2,0)
            L1 = (-self.N/2*t.log(self.alpha)).clone().detach().requires_grad_(True)
            L2 = self.alpha/2*(t.pow(self.norm_y,2).add(-self.alpha/self.beta*t.pow(self.norm_c,2)))
            L3 = t.trace(t.log(self.L))
            sum_L = t.add(L2,L3)
            self.likelihood = t.add(L1,sum_L)

        return self.likelihood

    
    def gradient(self, hp):
        """
        Gradient of the parameters of the network that are optimized through LBFGS
        hp :np.array((tot_par,))
            array that contains the weights of the network, alpha and beta
        """

        c = self.c_dim
        #Initialization
        if self.transfer==True and self.do_optimize==False:
            tot_par = self.T*2 +(c + self.dim_x)*D + 2*D**2 # (alpha and beta)*T + weights of the network
        else:
            # Net = 1( Transfer case, second pass of optimizing the ith task)
            # ABLR Simple
            tot_par = 2 +(c + self.dim_x)*D + 2*D**2 # (alpha and beta) + weights of the network

        g = np.zeros((tot_par))
        self.loss(hp).backward() # get the gradient retain_graph=True
        g0 = self.hidden1.weight.grad.data.numpy()
        g0 = g0.astype(np.float64) # issue with doubles and float
        g1 = self.hidden2.weight.grad.data.numpy()
        g1 = g1.astype(np.float64)
        g2 = self.hidden3.weight.grad.data.numpy()
        g2 = g2.astype(np.float64)

        if self.transfer==True and self.do_optimize==False:
            for i in range(0,self.T):
                if self.net == 1:
                    if i == self.i: continue
                g[2*D**2+(c + self.dim_x)*D + 2*i] = self.alpha_t[i].grad.data.numpy() #list of tensors for each task
                g[2*D**2+(c + self.dim_x)*D + 2*i +1] = self.beta_t[i].grad.data.numpy()
                self.alpha_t[i].grad.data.zero_()
                self.beta_t[i].grad.data.zero_()
        else:
            g[2*D**2+(c + self.dim_x)*D] = self.alpha.grad.data.numpy() # alpha gradient
            g[2*D**2+(c + self.dim_x)*D+1] = self.beta.grad.data.numpy() #beta gradient
            self.alpha.grad.data.zero_()
            self.beta.grad.data.zero_()
            
        g[0:(c + self.dim_x)*D]= g0.reshape(((c + self.dim_x)*D,))
        g[(c + self.dim_x)*D:D**2+(c + self.dim_x)*D] = g1.reshape((D**2,))
        g[D**2+(c + self.dim_x)*D:2*D**2+(c + self.dim_x)*D] = g2.reshape((D**2,))
        # zero gradient for the next optimization
        self.hidden1.weight.grad.data.zero_() 
        self.hidden2.weight.grad.data.zero_()
        self.hidden3.weight.grad.data.zero_()
        return g
    
    def optimize(self):
        """
        Optimize weights, alpha and beta with LBFGS
        """
        c= self.c_dim

        if self.transfer==True and self.do_optimize==False:
            tot_par = self.T*2 +(c + self.dim_x)*D + 2*D**2
        else:
            tot_par = 2 +(c + self.dim_x)*D + 2*D**2

        #Initial flattened array
        init = np.ones((tot_par))

        x00 = self.hidden1.weight.data.numpy()
        x01 = self.hidden2.weight.data.numpy()
        x02 = self.hidden3.weight.data.numpy()
        # initialize weight array and bounds for the weights
        init[0:(c + self.dim_x)*D]= x00.reshape(((c + self.dim_x)*D,))
        init[(c + self.dim_x)*D:D**2+(c + self.dim_x)*D] = x01.reshape((D**2,))
        init[D**2+(c + self.dim_x)*D:2*D**2+(c + self.dim_x)*D] = x02.reshape((D**2,))
        mybounds = [[None,None] for i in range(2*D**2+(c + self.dim_x)*D)]

        if self.transfer==True and self.do_optimize==False:
            for i in range(0,self.T*2,2):
                #Initialize alpha(initial value beta = 1)
                init[(c + self.dim_x)*D + 2*D**2 + i + 1] = 1e3
                #Bounds
                mybounds.append([1e-3,1e3])# alpha bounds
                mybounds.append([1e1,1e6])# beta bounds
        else:
            #Initialize alpha(second to last element)
            init[-1] = 1e3
            # Bounds
            mybounds.append([1e-3,1e3])
            mybounds.append([1e1,1e6])  
        hp = scipy.optimize.fmin_l_bfgs_b(self.loss, x0 = init, bounds=mybounds, fprime=self.gradient)

    
    def train(self,X, y, do_optimize=True):
        """
        First optimized the hyperparameters if do_optimize is True and then computes
        the posterior distribution of the weights.
        X: np.ndarray (N, D)
            Input data points.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true the hyperparameters are optimized for a single likelihood
            otherwise the hyperparameters are optimized for the sum of likelihoods
        self.coef : np.array(T,c)
            List of the coeficients(metadata) from all tasks
        self.i : int
            training task i
        (Net=2 the initial points are saved in X_init, Y_init)
        """
        c = self.c_dim
        y = y.reshape((y.shape[0],1))

        if not self.dataset: X = ((X - np.mean(X))/np.std(X)) # normalize input
        if self.ctx and not self.dataset:
            # concatenate input with coeficients
            coef_task = np.ones((X.shape[0],c))
            for i in range(X.shape[0]):
                coef_task[i] = self.coef[self.i]# coeficients for the specific task
            x_new = np.concatenate((X, coef_task), axis=1)
        elif self.ctx and self.dataset:
            # concatenate x with contextual information from the dataset
            meta_init = np.ones((X.shape[0],c))
            for j,x in enumerate(X): 
                key = str(x)
                meta_init[j-1] = self.f.metafeatures[key] # get the metafeatures corresponding to the x value
            x_new = np.concatenate((X,meta_init), axis=1).reshape((X.shape[0],c+self.dim_x))
        else: x_new = X

        if self.net == 2 and self.transfer==True:
            self.do_optimize=False
            # add the training points to the initial and optimize the sum of likelihoods(transfer cases)
            self.x_train = np.concatenate((self.X_init[self.i],x_new),axis=0)
            self.y_train = np.concatenate((self.Y_init[self.i],y),axis=0)
            self.x_train = t.reshape(t.from_numpy(self.x_train).float(),(self.X_init[self.i].shape[0]+x_new.shape[0],c + self.dim_x))
            self.y_train = t.reshape(t.from_numpy(self.y_train).float(),(self.Y_init[self.i].shape[0]+x_new.shape[0],1))
            #for the other tasks
            self.x = t.from_numpy(self.X_init).float()
            self.y = t.from_numpy(self.Y_init).float()
        else:
            # net=1 or ABLR simple
            x_new = t.from_numpy(x_new).reshape((X.shape[0],c+self.dim_x)).float()
            y = t.from_numpy(y)
            self.x = x_new
            self.y = y
            # for the incumbent
            self.x_t = self.x
            self.y_t = self.y 

        #self.cov = self.alpha*t.eye(D) + self.beta*t.matmul(t.t(self.Phi),self.Phi) # DxD
        #self.mean = self.beta*t.matmul(t.inverse(self.cov),t.matmul(t.t(self.Phi),self.y)) # Dx1
        self.optimize()
        
        
    def predict(self, X_test):
        """
        Returns the predictive mean and variance of the objective function at
        the given test points.
        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points
        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance
        """
        if self.net == 2: self.do_optimize=True

        if self.dataset: X_test = X_test.reshape((X_test.shape[1],))

        self.x = X_test.astype(np.float32)
        self.N = self.x.shape[0]
        c = self.c_dim
        #Initial flattened array
        tot_par = 2 +(c + self.dim_x)*D + 2*D**2
        hp = np.ones((tot_par))
        # get the weights of the trained network
        x00 = self.hidden1.weight.data.numpy()
        x01 = self.hidden2.weight.data.numpy()
        x02 = self.hidden3.weight.data.numpy()

        hp[0:(c + self.dim_x)*D]= x00.reshape(((c + self.dim_x)*D,))
        hp[(c + self.dim_x)*D:D**2+(c + self.dim_x)*D] = x01.reshape((D**2,))
        hp[D**2+(c + self.dim_x)*D:2*D**2+(c + self.dim_x)*D] = x02.reshape((D**2,))
        if self.net == 2 and self.transfer == True :
            hp[-2] = self.alpha_t[self.i]
            hp[-1] = self.beta_t[self.i]
        else:
            hp[-2] = self.alpha
            hp[-1] = self.beta

        if self.ctx and not self.dataset:
            # concatenate input with coeficients
            cf = np.reshape(self.coef[self.i],(1,3))
            self.x = t.from_numpy(np.concatenate((self.x, cf), axis=1)).float()
        elif self.ctx and self.dataset:
            meta_init = self.f.metafeatures[str(self.x)]
            self.x = self.x.reshape((1,self.x.shape[0]))
            meta_init = meta_init.reshape((1,meta_init.shape[0]))
            self.x = t.from_numpy(np.concatenate((self.x,meta_init), axis=1).reshape((self.x.shape[0],c+self.dim_x))).float()
        elif self.dataset: self.x = t.from_numpy(self.x).reshape((1,self.x.shape[0]))
        else: self.x = t.from_numpy(self.x)
            
        if self.f==None :
            f = Quadratic_function(self.coef[self.i])
            self.y = t.tensor(f.call(X_test), requires_grad=True).reshape((1,))
        else : self.y = t.tensor(self.f.call(X_test), requires_grad=True).reshape((1,))
        self.loss(hp) # get the new feature matrix

        m = t.matmul(t.matmul(t.t(self.c), self.L_inv), t.t(self.Phi))# predictive mean
        m = (self.alpha/self.beta)*m.reshape((m.shape[1],1)) # Nx1
        m =m.detach().numpy()
        assert(m.shape == (1,1))
        v = t.matmul(self.L_inv,t.t(self.Phi))# predictive variance
        v = 1/self.beta*t.pow(t.norm(v),2) + 1/self.alpha 
        v =v.detach().numpy()
        assert(v.shape == (1,))

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """
        c = self.c_dim
        if self.net == 2 and self.transfer == True :
            y = self.y_train.detach().numpy()
            x = self.x_train.detach().numpy()
        else : # net=1 or ABLR simple
            y = self.y_t.detach().numpy()
            x = self.x_t.detach().numpy()
        best_idx = np.argmin(y)
        if self.ctx:
            x_inc = x[best_idx][:c] # take only the x value because this matrix also contains the coefficients
        else:
            x_inc = x[best_idx]
        return x_inc, y[best_idx]
