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
import theano.tensor as tt

num_samples = 100
warmup_steps = 100

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

def test_pyro():
    # define data and model
    train_x = torch.linspace(0, 1, 10).double()
    train_y = torch.sin(train_x * (2 * math.pi)).double() + torch.randn(train_x.size()).double() * 0.1
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.0, 9.0), "lengthscale")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(0, 4), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.0, 0.25), "noise")

    model.double()
    likelihood.double()

    # define pyro primitive
    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            loss = mll(output, y)*y.shape[0]
        pyro.factor("gp_mll", loss)

    # initialize model parameters
    model.mean_module.constant.data.fill_(0.0)
    model.covar_module.outputscale = 0.5**2
    model.covar_module.base_kernel.lengthscale = 1
    model.likelihood.noise = 0.05**2

    initial_params =  {'mean_module.mean_prior': torch.DoubleTensor([model.mean_module.constant.data]),\
        'covar_module.base_kernel.lengthscale_prior':  torch.DoubleTensor([model.covar_module.base_kernel.raw_lengthscale.data]),\
        'covar_module.outputscale_prior': torch.DoubleTensor([model.covar_module.raw_outputscale.data]),\
        'likelihood.noise_prior': torch.DoubleTensor([model.likelihood.raw_noise.data])}

    # transforms = {'mean_module.mean_prior': model.mean_module.mean_prior.transform,\
    #     'covar_module.base_kernel.lengthscale_prior':  model.covar_module.base_kernel.lengthscale_prior.transform,\
    #     'covar_module.outputscale_prior': model.covar_module.outputscale_prior.transform,\
    #     'likelihood.noise_prior': model.likelihood.noise_prior.transform}

    # define nuts and set up
    
    args = (train_x, train_y)
    kwargs = {}
    nuts_kernel = NUTS(pyro_model)
    nuts_kernel.setup(0, *args, **kwargs)
    
    # evaluate potential function several times
    for i in range(10):
        print(nuts_kernel.potential_fn(initial_params).item())


def main():

    train_x = torch.linspace(0, 1, 100).double()
    train_y = torch.sin(train_x * (2 * math.pi)).double() + torch.randn(train_x.size()).double() * 0.1
    # data = np.genfromtxt("example.csv")
    # train_x = torch.tensor(data[0])
    # train_y = torch.tensor(data[1])

    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = ExactGPModel(train_x, train_y, likelihood)

    model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
    model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.0, 9.0), "lengthscale")
    # model.covar_module.base_kernel.register_prior("period_length_prior", UniformPrior(0.0, 4.0), "period_length")
    model.covar_module.register_prior("outputscale_prior", UniformPrior(0, 4), "outputscale")
    likelihood.register_prior("noise_prior", UniformPrior(0.0, 0.25), "noise")

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            loss = mll(output, y)*y.shape[0]
        pyro.factor("gp_mll", loss)

    model.mean_module.constant.data.fill_(0.0)
    model.covar_module.outputscale = 0.5**2
    model.covar_module.base_kernel.lengthscale = 1
    model.likelihood.noise = 0.05**2

    model.double()
    likelihood.double()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.LBFGS(model.parameters(), history_size=10, max_iter=4)

    model.train()
    likelihood.train()
    current_ll = 0
    current_state = model.state_dict()
    training_iterations = 0
    for i in range(training_iterations):
        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                loss = -mll(output, train_y)*train_x.shape[0]
            print('Iter %d/%d - LL: %.3f' % (i + 1, training_iterations, -loss.item()))
            loss.backward()
            return loss
        
        optimizer.step(closure)


    # print(model.mean_module.constant.data)
    # print(np.sqrt(model.covar_module.outputscale.detach().numpy()))
    # print(model.covar_module.base_kernel.lengthscale)
    # print(np.sqrt(model.likelihood.noise.detach().numpy()))
    

    initial_params =  {'mean_module.mean_prior': torch.DoubleTensor([model.mean_module.constant.data]),\
        'covar_module.base_kernel.lengthscale_prior':  torch.DoubleTensor([model.covar_module.base_kernel.raw_lengthscale.data]),\
        'covar_module.outputscale_prior': torch.DoubleTensor([model.covar_module.raw_outputscale.data]),\
        'likelihood.noise_prior': torch.DoubleTensor([model.likelihood.raw_noise.data])}

    def potential_fn(z):
        model.mean_module.constant.data.fill_(z['mean_module.mean_prior'].item())
        model.covar_module.base_kernel.raw_lengthscale.data.fill_(z["covar_module.base_kernel.lengthscale_prior"].item())
        model.covar_module.raw_outputscale.data.fill_(z["covar_module.outputscale_prior"].item())
        model.likelihood.raw_noise.data.fill_(z['likelihood.noise_prior'].item())
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            output = model(train_x)
            loss = -mll(output, train_y)*train_y.shape[0]
        return loss


    nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
    hmc_kernel = HMC(pyro_model, step_size=0.1, num_steps=10, adapt_step_size=True,\
             init_strategy=pyro.infer.autoguide.initialization.init_to_median(num_samples=20))
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)#, initial_params=initial_params)

    return model, likelihood, mll, mcmc_run, train_x, train_y

    # with pm.Model() as m:
    #     c = pm.Uniform("c", lower=-1, upper=1)
    #     ls = pm.Uniform("ls", lower=0, upper=9)
    #     os = pm.Uniform("os", lower=0, upper=1)
    #     noise = pm.Uniform("noise", lower=0, upper=1)
    #     model.mean_module.constant.data.fill_(tt.as_tensor_variable(c))
    #     model.covar_module.base_kernel.lengthscale = ls
    #     model.covar_module.outputscale = os
    #     model.likelihood.noise  = noise

    #     with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
    #         output = model(x)


    #     y_obs = pm.MvNormal("y_obs", mu=output.mean, cov=output.covariance, observed=y)

    #     start = pm.find_MAP()
    #     step = pm.Slide()
    #     trace = pm.sample(100, step, start)
    # return


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
    # train(mcmc_run, train_x, train_y)
    # mcmc_run.run(train_x, train_y)
    mcmc_run = pickle.load(open("results/test_mcmc.pkl",'rb'))
    print(mcmc_run.diagnostics())
    mcmc_samples = mcmc_run.get_samples()
    param_list = ["likelihood.noise_prior", "covar_module.outputscale_prior",
    "covar_module.base_kernel.lengthscale_prior","mean_module.mean_prior"]
    labels = ["noise", "os","ls","mean"]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
         samples = mcmc_samples[param_list[i]].numpy().reshape(-1)
         if i<=1:
             samples = np.sqrt(samples)
         sns.distplot(samples, ax=axes[int(i/2), int(i%2)])
         axes[int(i/2)][int(i%2)].legend([labels[i]])
    plt.show()

    model.pyro_load_from_samples(mcmc_run.get_samples())
    model.train()
    likelihood.train()
    test_x = torch.linspace(0, 1, 100).unsqueeze(-1)
    test_y = torch.sin(test_x * (2 * math.pi))
    expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
    expanded_train_y = train_y.unsqueeze(0).repeat(num_samples, 1)
    output = model(expanded_test_x)

    losses = None
    for i in range(1):
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            losses = mll(output, expanded_train_y)*train_y.shape[0]

    sns.distplot(losses.detach().numpy())
    plt.show()

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
