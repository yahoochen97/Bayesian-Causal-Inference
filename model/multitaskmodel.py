import torch
import gpytorch
from torch.nn import ModuleList


class MultitaskGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y,likelihood):
        '''
        Inputs:
            - train_x: 2d tensor of shape [(N_tr+N_co)*T-N_tr*(T-T0+1)]*(d+1), last column is time
            - train_y: 1d tensor of shape (N_tr+N_co)*T-N_tr*(T-T0+1)
            - likelihood: gpytorch.likelihood object
        '''
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_task = 2 # treatment/control
        self.d = list(train_x[0].shape)[1] - 1 # dim of covariates
        self.mean_module = ModuleList([gpytorch.means.LinearMean(self.d+1) for _ in range(self.num_task)])
        self.x_covar_module = gpytorch.kernels.RBFKernel(active_dims=tuple([i for i in range(self.d)]))
        self.t_covar_module = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([self.d])) \
            + gpytorch.kernels.RBFKernel(active_dims=torch.tensor([self.d]))
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=self.num_task, rank=1)

    def forward(self, x, i):
        mean_x = None
        for j,idx in enumerate(i):
            if j==0:
                mean_x = self.mean_module[idx](x[j].reshape(1,-1))
            else:
                mean_x = torch.cat([mean_x, self.mean_module[idx](x[j].reshape(1,-1))])

        covar_x = self.x_covar_module(x)
        covar_t = self.t_covar_module(x)
        covar_i = self.task_covar_module(i)
        covar = covar_x.mul(covar_i) + covar_t.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

