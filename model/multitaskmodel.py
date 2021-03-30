import torch
import gpytorch
from torch.nn import ModuleList
import json
import numpy as np
from model.customizedkernel import myIndexKernel, constantKernel, myIndicatorKernel


class MultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, X_max_v, likelihood,num_mixtures=4):
        '''
        Inputs:
            - train_x: 2d tensor of shape [(N_tr+N_co)*T-N_tr*(T-T0+1)]*(d+1), last column is time
            - train_y: 1d tensor of shape (N_tr+N_co)*T-N_tr*(T-T0+1)
            - likelihood: gpytorch.likelihood object
        '''
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        # load configurations
        with open('model/conf.json') as f:
            configs = json.load(f)
            sigma_a = configs["sigma_a"]
            sigma_b = configs["sigma_b"]
            sigma_L = configs["sigma_L"]
            sigma_beta = configs["sigma_beta"]

        # define priors
        outputscale_prior = [gpytorch.priors.GammaPrior(concentration=1,rate=2),\
            gpytorch.priors.GammaPrior(concentration=1,rate=5)]
        lengthscale_prior = [gpytorch.priors.GammaPrior(concentration=3,rate=1/5),\
            gpytorch.priors.GammaPrior(concentration=1,rate=1/2)]
        rho_prior = gpytorch.priors.UniformPrior(-1, 1)
            
        # self.T = T
        # self.T0 = T0
        self.num_task = 2 # treatment/control
        self.X_max_v = X_max_v
        self.d = list(train_x[0].shape)[1] - 1 # dim of covariates
        # self.x_mean_module = gpytorch.means.LinearMean(self.d, bias=False)
        # self.mean_module = ModuleList([gpytorch.means.LinearMean(self.d+1) for _ in range(self.num_task)])
        # self.mean_module = ModuleList([gpytorch.means.ConstantMean() for _ in range(self.num_task)])
        self.i_mean_module = gpytorch.means.LinearMean(1, bias=True)
        # self.x_covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=self.d, \
        #     active_dims=tuple([i for i in range(self.d)]))
        self.x_covar_module = ModuleList([constantKernel(num_tasks=v+1) for v in X_max_v])
        # self.t_covar_module = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([self.d])) \
        #     + gpytorch.kernels.RBFKernel(active_dims=torch.tensor([self.d]))
        # self.t_covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, \
        #     active_dims=torch.tensor([self.d]))
        self.t_covar_module = ModuleList(gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d]), lengthscale_prior=lengthscale_prior[i])\
                 ,outputscale_prior=outputscale_prior[i]) for i in range(1))
        # self.c_covar_module = constantKernel(num_tasks=2)
        # self.indicator_module = myIndicatorKernel(num_tasks=2)
        self.indicator_module = ModuleList([myIndicatorKernel(num_tasks=v+1) for v in X_max_v])
        
        # sd_prior = gpytorch.priors.SmoothedBoxPrior(np.exp(-1) , np.exp(1))
        # cov_prior = gpytorch.priors.LKJCovariancePrior(n=self.num_task,\
        #      eta=1, sd_prior=sd_prior)
        # cov_prior = gpytorch.priors.MultivariateNormalPrior(torch.zeros(self.num_task),\
        #     torch.eye(self.num_task))
        
        self.task_covar_module = myIndexKernel(num_tasks=self.num_task, rho_prior=rho_prior)

        # self.task_covar_module = gpytorch.kernels.ScaleKernel(\
        #     gpytorch.kernels.IndexKernel(num_tasks=self.num_task, rank=1))
        # prior=cov_prior
        # var_constraint=gpytorch.constraints.Interval(0,1)
        # self.task_covar_module.register_prior("IndexKernelPrior", cov_prior, "covar_factor")

        # self.x_mean_module.register_prior("weights_prior", \
        #     gpytorch.priors.MultivariateNormalPrior(torch.zeros(self.d), torch.eye(self.d)), "weights")
        # for i in range(self.num_task):
        #     self.t_mean_module[i].register_prior("weights_prior", gpytorch.priors.NormalPrior(0.0, sigma_a), "weights")
        #     self.t_mean_module[i].register_prior("bias_prior", gpytorch.priors.NormalPrior(np.log(0.1), sigma_b), "bias")

    def forward(self, x, i):
        # mean_x = None
        # for j,idx in enumerate(i):
        #     # tmp = self.x_mean_module(x[j,:self.d].reshape(1,-1)) +\
        #     #      self.t_mean_module[idx](x[j,self.d].reshape(1,-1)) 
        #     tmp = self.mean_module[idx](x[j].reshape(1,-1)).double()
        #     if j==0:
        #         mean_x = tmp
        #     else:
        #         mean_x = torch.cat([mean_x, tmp])
        mean_i = self.i_mean_module(i.double())

        # covar_x = self.x_covar_module(x)
        # self.t_covar_module.base_kernel.lengthscale=1
        # self.t_covar_module.outputscale=1
        # i= torch.tensor([0,0,1,1]).reshape((-1,1))
        # x= torch.tensor([0,5,0,1]).reshape((-1,1))
        # self.c_covar_module.c2.data.fill_(0.5)
        # self.task_covar_module.rho.data.fill_(0.5)

        # covar_x = self.x_covar_module(x)
        covar_t = self.t_covar_module[0](x) # + self.t_covar_module[1](x)
        covar_i = self.task_covar_module(i)
        covar = covar_t.mul(covar_i)
        for j in range(len(self.X_max_v)):
        # for j in range(1):
            covar_c = self.x_covar_module[j](x[:,j].long())
            indicator = self.indicator_module[j](x[:,j].long())
            covar += covar_c.mul(indicator) # + covar_x.mul(covar_i) 

        return gpytorch.distributions.MultivariateNormal(mean_i.double(), covar.double())

