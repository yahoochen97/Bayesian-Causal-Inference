import torch
import gpytorch
from torch.nn import ModuleList
import json
import numpy as np
from model.customizedkernel import myIndexKernel, constantKernel, myIndicatorKernel
from model.customizedkernel import ConstantVectorMean, DriftScaleKernel, DriftIndicatorKernel

class MultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, X_max_v, likelihood, MAP=True):
        '''
        Inputs:
            - train_x:
            - train_y:
            - likelihood:
        '''
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)

        # define priors
        outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        lengthscale_prior = gpytorch.priors.GammaPrior(concentration=4,rate=1/5)
        rho_prior = gpytorch.priors.UniformPrior(-1, 1)
        unit_outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        unit_lengthscale_prior = gpytorch.priors.GammaPrior(concentration=4,rate=1/5)
        drift_outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=20)
        drift_lengthscale_prior = gpytorch.priors.GammaPrior(concentration=5,rate=1/5)
        weekday_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        day_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        
        # treatment/control groups
        self.num_groups = 2 
        self.num_units = len(train_x[:,-3].unique())

        # categoritcal features: group/weekday/day/unit id
        self.X_max_v = X_max_v
        # dim of covariates
        self.d = list(train_x.shape)[1] - 1

        # same mean of unit bias for all units, could extend this to be unit-dependent
        # self.unit_mean_module = gpytorch.means.ConstantMean()
        self.unit_mean_module =  ConstantVectorMean(d=self.num_units)
        self.group_mean_module = ConstantVectorMean(d=self.num_groups)

        # marginalize weekday/day/unit id effects
        self.x_covar_module = ModuleList([constantKernel(num_tasks=v+1) for v in self.X_max_v])

        # self.x_covar_module = ModuleList([constantKernel(num_tasks=X_max_v[0]+1, prior=weekday_prior),
        #         constantKernel(num_tasks=X_max_v[1]+1, prior=day_prior),
        #         constantKernel(num_tasks=X_max_v[2]+1)])

        # group-level time trend
        self.group_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
                active_dims=torch.tensor([self.d]),\
                     lengthscale_prior=lengthscale_prior if MAP else None),\
                    outputscale_prior=outputscale_prior if MAP else None)

        # indicator covariances
        self.x_indicator_module = ModuleList([myIndicatorKernel(num_tasks=v+1) for v in X_max_v])
        self.group_index_module = myIndexKernel(num_tasks=self.num_groups,\
             rho_prior=rho_prior if MAP else None)

        # unit-level zero-meaned time trend
        self.unit_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d]),\
            lengthscale_prior=unit_lengthscale_prior if MAP else None),\
            outputscale_prior=unit_outputscale_prior if MAP else None)

        self.unit_indicator_module = myIndicatorKernel(num_tasks=len(train_x[:,-3].unique()))

        # drift process for treatment effect
        self.drift_t_module = DriftScaleKernel(gpytorch.kernels.RBFKernel(\
                active_dims=torch.tensor([self.d]),\
                lengthscale_prior=drift_lengthscale_prior if MAP else None),\
                outputscale_prior=drift_outputscale_prior if MAP else None)
        self.drift_indicator_module = DriftIndicatorKernel(num_tasks=self.num_groups)

    def forward(self, x):
        if len(x.shape)==2:
            group = x[:,-2].reshape((-1,1)).long()
            units = x[:,-3].reshape((-1,1)).long()
            ts = x[:,-1]
        else:
            group = x[0,:,-2].reshape((-1,1)).long()
            units = x[0,:,-3].reshape((-1,1)).long()
            ts = x[0,:,-1]

        # only non-zero unit-level mean
        # mu = self.unit_mean_module(x)
        mu = self.group_mean_module(group) + self.unit_mean_module(units) 
        mu = mu.reshape(-1,)
        
        # covariance for time trends
        covar_group_t = self.group_t_covar_module(x)
        covar_group_index = self.group_index_module(group)
        covar_unit_t = self.unit_t_covar_module(x)
        covar_unit_indicator = self.unit_indicator_module(units)
        covar = covar_group_t.mul(covar_group_index) + covar_unit_t.mul(covar_unit_indicator)

        if self.drift_t_module.T0 is not None:
            covar_drift_indicator = self.drift_indicator_module(group)
            covar_drift_t = self.drift_t_module(x)
            covar += covar_drift_t.mul(covar_drift_indicator)

        # marginalize weekday/day/unit id effects
        for j in range(len(self.X_max_v)):
            if len(x.shape)==2:
                covar_c = self.x_covar_module[j](x[:,j].long())
                indicator = self.x_indicator_module[j](x[:,j].long())
            else:
                # batch realization
                num_samples = x.shape[0]
                n = x.shape[1]
                tmp = x[:,:,j].reshape(num_samples,n).long()
                covar_c = self.x_covar_module[j].forward(tmp, tmp)
                tmp = x[:,:,j].reshape(num_samples,n,1).long()
                indicator = self.x_indicator_module[j].forward(tmp, tmp)
            covar += indicator.mul(covar_c)

        return gpytorch.distributions.MultivariateNormal(mu.double(), covar.double())


class PresentationModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, X_max_v, likelihood, MAP=True):
        super(PresentationModel, self).__init__(train_x, train_y, likelihood)

        # define priors
        outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        lengthscale_prior = gpytorch.priors.GammaPrior(concentration=3,rate=1/5)
        rho_prior = gpytorch.priors.UniformPrior(-1, 1)
        unit_outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        unit_lengthscale_prior = gpytorch.priors.GammaPrior(concentration=4,rate=1/5)
        weekday_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        day_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        
        # treatment/control groups
        self.num_groups = 2 
        self.num_units = len(train_x[:,-3].unique())

        # categoritcal features: group/weekday/day/unit id
        self.X_max_v = X_max_v
        # dim of covariates
        self.d = list(train_x.shape)[1] - 1

        # same mean of unit bias for all units, could extend this to be unit-dependent
        self.unit_mean_module =  ConstantVectorMean(d=self.num_units)
        self.group_mean_module = ConstantVectorMean(d=self.num_groups)

        # marginalize weekday/day/unit id effects
        # self.x_covar_module = ModuleList([constantKernel(num_tasks=v+1) for v in self.X_max_v])

        # group-level time trend
        self.group_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
                active_dims=torch.tensor([self.d]),\
                     lengthscale_prior=lengthscale_prior if MAP else None),\
                    outputscale_prior=outputscale_prior if MAP else None)

        # indicator covariances
        # self.x_indicator_module = ModuleList([myIndicatorKernel(num_tasks=v+1) for v in X_max_v])
        self.group_index_module = myIndexKernel(num_tasks=self.num_groups,\
             rho_prior=rho_prior if MAP else None)

        # unit-level zero-meaned time trend
        self.unit_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d]),\
            lengthscale_prior=unit_lengthscale_prior if MAP else None),\
                outputscale_prior=unit_outputscale_prior if MAP else None)

        self.unit_indicator_module = myIndicatorKernel(num_tasks=len(train_x[:,-3].unique()))

    def forward(self, x):
        group = x[:,-2].reshape((-1,1)).long()
        units = x[:,-3].reshape((-1,1)).long()

        # only non-zero unit-level mean
        mu = self.unit_mean_module(units) + self.group_mean_module(group)
        mu = mu.reshape(-1,)
        
        # covariance for time trends
        covar_group_t = self.group_t_covar_module(x)
        covar_group_index = self.group_index_module(group)
        covar_unit_t = self.unit_t_covar_module(x)
        covar_unit_indicator = self.unit_indicator_module(units)
        covar = covar_group_t.mul(covar_group_index) + covar_unit_t.mul(covar_unit_indicator)

        # marginalize weekday/day/unit id effects
        # for j in range(len(self.X_max_v)):
        #     covar_c = self.x_covar_module[j](x[:,j].long())
        #     indicator = self.x_indicator_module[j](x[:,j].long())
        #     covar += indicator.mul(covar_c)

        return gpytorch.distributions.MultivariateNormal(mu.double(), covar.double())

