import math
import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt
import os
import dill as pickle

from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():
    smoke_test = ('CI' in os.environ)
    num_samples = 2 if smoke_test else 100
    warmup_steps = 2 if smoke_test else 200

    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2

    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)

    model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.001, 1.0), "lengthscale")
    model.covar_module.base_kernel.register_prior("period_length_prior", UniformPrior(0.05, 2.5), "period_length")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.05, 0.3), "noise")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x)
        loss = mll.pyro_factor(output, y)
        return y

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=smoke_test)
    return mcmc_run, train_x, train_y

def train(mcmc_run, train_x, train_y):
    mcmc_run.run(train_x, train_y)
    pickle.dump(mcmc_run, open("results/test_mcmc.pkl", "wb"))

if __name__ == "__main__":
    mcmc_run, train_x, train_y = main()
    # train(mcmc_run, train_x, train_y)

    mcmc_run = pickle.load(open("results/test_mcmc.pkl",'rb'))

    print(mcmc_run.diagnostics())

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
