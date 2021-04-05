import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from matplotlib import pyplot as plt
import os
import dill as pickle
from matplotlib import pyplot as plt
import scipy.special as sps 
import seaborn as sns
import pymc3 as pm
import sampyl as smp
import numpy as np

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
    smoke_test = ('CI' in os.environ)
    num_samples = 2 if smoke_test else 500
    warmup_steps = 2 if smoke_test else 500

    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.1

    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)

    model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 4), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(0.01, 4), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.0001, 0.04), "noise")
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def logp(c, ls, os, noise):
        if type(c) is not float:
            c = c._value
        if type(ls) is not float:
            ls = ls._value
        if type(os) is not float:
            os = os._value
        if type(noise) is not float:
            noise = noise._value
        #     c = params[0]._value
        #     ls = params[1]._value
        #     os = params[2]._value
        #     noise = params[3]._value
        # else:
        #     c = params[0]
        #     ls = params[1]
        #     os = params[2]
        #     noise = params[3]
        model.mean_module.constant.data.fill_(torch.tensor(c))
        model.covar_module.base_kernel.raw_lengthscale.data.fill_(ls)
        model.covar_module.raw_outputscale.data.fill_(os)
        model.likelihood.raw_noise.data.fill_(noise)
        output = model(train_x)
        ll = mll(output, train_y)*train_x.shape[0]
        return ll.detach().numpy()

    model.mean_module.constant.data.fill_(0.0)
    model.covar_module.base_kernel.lengthscale = 1
    model.covar_module.outputscale = 0.5**2
    model.likelihood.noise = 0.05**2

    start = {'params': np.array([model.mean_module.constant.item(),\
          model.covar_module.base_kernel.raw_lengthscale.item(),\
              model.covar_module.raw_outputscale.item(),  model.likelihood.raw_noise.detach().item()])}

    start = {'c': model.mean_module.constant.item(),\
         'ls':  model.covar_module.base_kernel.raw_lengthscale.item(),\
        'os': model.covar_module.raw_outputscale.item(),\
        'noise': model.likelihood.raw_noise.detach().item()
    }

    hmc = smp.Hamiltonian(logp, start, step_size=0.1, n_steps=10)
    nuts = smp.NUTS(logp, start)
    chain = hmc.sample(100, burn=10)

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


    # model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    # model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
    # model.covar_module.base_kernel.register_prior("period_length_prior", UniformPrior(0.05, 2.5), "period_length") 
    # model.covar_module.register_prior("outputscale_prior", UniformPrior(0.5, 2), "outputscale")
    # likelihood.register_prior("noise_prior", UniformPrior(0.001, 0.2), "noise")


    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x)
        loss = mll.pyro_factor(output, y)
        return loss

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
    hmc_kernel = HMC(pyro_model, step_size=0.1, num_steps=10, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=smoke_test)
    return mcmc_run, train_x, train_y

def train(mcmc_run, train_x, train_y):
    mcmc_run.run(train_x, train_y)
    pickle.dump(mcmc_run, open("results/test_mcmc.pkl", "wb"))


if __name__ == "__main__":

    mcmc_run, train_x, train_y = main()
    train(mcmc_run, train_x, train_y)
    mcmc_run = pickle.load(open("results/test_mcmc.pkl",'rb'))
    mcmc_samples = mcmc_run.get_samples()
    param_list = ["likelihood.noise_prior", "covar_module.outputscale_prior",
    "covar_module.base_kernel.lengthscale_prior","mean_module.mean_prior"]
    labels = ["noise", "os","ls","mean"]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
         samples = mcmc_samples[param_list[i]].numpy().reshape(-1)
         sns.distplot(samples, ax=axes[int(i/2), int(i%2)])
         axes[int(i/2)][int(i%2)].legend([labels[i]])
    plt.show()

    # model.pyro_load_from_samples(mcmc_run.get_samples())
    # model.eval()
    # test_x = torch.linspace(0, 1, 101).unsqueeze(-1)
    # test_y = torch.sin(test_x * (2 * math.pi))
    # expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
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
