import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from pyro.distributions import Normal
from matplotlib import pyplot as plt
import os
import dill as pickle
from matplotlib import pyplot as plt
import scipy.special as sps 
import seaborn as sns
import pymc3 as pm
import sampyl as smp
import numpy as np
import theano.tensor as tt

num_samples = 500
warmup_steps = 500

torch.set_default_tensor_type(torch.DoubleTensor)

torch.multiprocessing.set_sharing_strategy('file_system')

from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():

    train_x = torch.linspace(0, 1, 5000).double()
    train_y = torch.sin(train_x * (2 * math.pi)).double() + torch.randn(train_x.size()).double() * 0.1

    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)

    # model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    # model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.0, 9.0), "lengthscale")
    # # model.covar_module.base_kernel.register_prior("period_length_prior", UniformPrior(0.0, 4.0), "period_length")
    # model.covar_module.register_prior("outputscale_prior", UniformPrior(0, 4), "outputscale")
    # likelihood.register_prior("noise_prior", UniformPrior(0.0, 0.25), "noise")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        priors= {
            'covar_module.base_kernel.raw_lengthscale': Normal(0, 2).expand([1, 1]),
            'covar_module.raw_outputscale': Normal(0, 2),
            'likelihood.noise_covar.raw_noise': Normal(0, 2).expand([1]),
            'mean_module.constant': Normal(0, 2),
        }
        fn = pyro.random_module("model", model, prior=priors)
        sampled_model = fn()
        
        output = sampled_model.likelihood(sampled_model(x))
        pyro.sample("obs", output, obs=y)

    # model.mean_module.constant.data.fill_(0.0)
    # model.covar_module.outputscale = 0.5**2
    # model.covar_module.base_kernel.lengthscale = 1
    # model.likelihood.noise = 0.05**2

    model.double()
    likelihood.double()

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True, jit_compile=False)
    hmc_kernel = HMC(pyro_model, step_size=0.1, num_steps=10, adapt_step_size=True,\
             init_strategy=pyro.infer.autoguide.initialization.init_to_median(num_samples=20))
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)#, initial_params=initial_params)

    return model, likelihood, mll, mcmc_run, train_x, train_y

    labels = ["c", "ls","os","noise"]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
        if i == 0:
            samples = getattr(chain, labels[i]).reshape(-1)
        else:
            samples = 1/(1+np.exp(-1*getattr(chain, labels[i]).reshape(-1)))
        sns.distplot(samples, ax=axes[int(i/2), int(i%2)])
        axes[int(i/2)][int(i%2)].legend([labels[i]])
    plt.show()
    pickle.dump(chain, open("results/test_mcmc.pkl", "wb"))
    return


def train(mcmc_run, train_x, train_y):
    mcmc_run.run(train_x, train_y)
    pickle.dump(mcmc_run, open("results/test_mcmc.pkl", "wb"))


if __name__ == "__main__":
    model, likelihood, mll, mcmc_run, train_x, train_y = main()
    train(mcmc_run, train_x, train_y)
    # mcmc_run.run(train_x, train_y)
    mcmc_run = pickle.load(open("results/test_mcmc.pkl",'rb'))
    # print(mcmc_run.diagnostics())
    mcmc_samples = mcmc_run.get_samples()
    param_list = ["model$$$likelihood.noise_covar.raw_noise", "model$$$covar_module.raw_outputscale",
    "model$$$covar_module.base_kernel.raw_lengthscale","model$$$mean_module.constant"]
    labels = ["noise", "os","ls","mean"]
    tranforms = [model.likelihood.noise_covar.raw_noise_constraint.transform,
    model.covar_module.raw_outputscale_constraint.transform,
    model.covar_module.base_kernel.raw_lengthscale_constraint.transform]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
         samples = mcmc_samples[param_list[i]].numpy().reshape(-1)
         if i<3:
             samples = [tranforms[i](torch.DoubleTensor([s])).item() for s in samples]
         sns.distplot(samples, ax=axes[int(i/2), int(i%2)])
         axes[int(i/2)][int(i%2)].legend([labels[i]])
    plt.show()

    # model.pyro_load_from_samples(mcmc_run.get_samples())
    # model.train()
    # likelihood.train()
    # test_x = torch.linspace(0, 1, 100).unsqueeze(-1)
    # test_y = torch.sin(test_x * (2 * math.pi))
    # expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
    # expanded_train_y = train_y.unsqueeze(0).repeat(num_samples, 1)
    # output = model(expanded_test_x)

    # with torch.no_grad():
    #     # Initialize plot
    #     f, ax = plt.subplots(1, 1, figsize=(4, 3))

    #     # Plot training data as black stars
    #     ax.plot(train_x.numpy(), train_y.numpy(), 'k*', zorder=10)

    #     for i in range(min(num_samples, 25)):
    #         # Plot predictive means as blue line
    #         ax.plot(test_x.numpy(), output.mean[i].detach().numpy(), 'b', linewidth=0.3)

    #     # Shade between the lower and upper confidence bounds
    #     # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #     ax.set_ylim([-3, 3])
    #     ax.legend(['Observed Data', 'Sampled Means'])

    # import seaborn as sns
    # data = mcmc_run.get_samples()["covar_module.base_kernel.lengthscale_prior"].numpy().reshape(-1)

    # sns.kdeplot(data=data)
