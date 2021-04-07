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
        # define priors
        outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        lengthscale_prior = gpytorch.priors.GammaPrior(concentration=3,rate=1/5)
        rho_prior = gpytorch.priors.UniformPrior(-1, 1)
        lengthscale_prior = gpytorch.priors.UniformPrior(0.1, 60)
        outputscale_prior = gpytorch.priors.UniformPrior(0, 0.04)
            
        self.num_task = 2 # treatment/control
        self.X_max_v = X_max_v
        self.d = list(train_x.shape)[1] - 1 # dim of covariates
        self.g_mean_module = gpytorch.means.LinearMean(1, bias=True)
        self.x_covar_module = ModuleList([constantKernel(num_tasks=v+1) for v in X_max_v])
        self.t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d]), lengthscale_prior=lengthscale_prior),outputscale_prior=outputscale_prior)

        self.indicator_module = ModuleList([myIndicatorKernel(num_tasks=v+1) for v in X_max_v])
        self.task_covar_module = myIndexKernel(num_tasks=self.num_task, rho_prior=rho_prior)
        self.unit_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d]), lengthscale_prior=lengthscale_prior),outputscale_prior=outputscale_prior)
        self.unit_indicator_module = myIndicatorKernel(num_tasks=len(train_x[:,-2].unique()))

    def forward(self, x):
        if len(x.shape)==2:
            group = x[:,0].reshape((-1,1)).long()
            units = x[:,-2].reshape((-1,1)).long()
        else:
            group = x[0,:,0].reshape((-1,1)).long()
            units = x[:,:,-2].reshape((-1,1)).long()
        mean_i = self.g_mean_module(group.double())

        covar_t = self.t_covar_module(x)
        covar_g = self.task_covar_module(group)
        covar_unit_t = self.unit_t_covar_module(x)
        covar_i = self.unit_indicator_module(units)
        covar = covar_t.mul(covar_g) + covar_unit_t.mul(covar_i)
        for j in range(len(self.X_max_v)):
            if len(x.shape)==2:
                covar_c = self.x_covar_module[j](x[:,j].long())
                indicator = self.indicator_module[j](x[:,j].long())
            else:
                num_samples = x.shape[0]
                n = x.shape[1]
                covar_c = self.x_covar_module[j].forward(x[:,:,j].reshape(num_samples,n).long(),\
                    x[:,:,j].reshape(num_samples,n).long())
                indicator = self.indicator_module[j].forward(x[:,:,j].reshape(num_samples,n,1).long(),\
                x[:,:,j].reshape(num_samples,n,1).long())
            covar += indicator.mul(covar_c)

        return gpytorch.distributions.MultivariateNormal(mean_i.double(), covar.double())

