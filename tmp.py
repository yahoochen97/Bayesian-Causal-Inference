import math
import torch
import gpytorch
from gpytorch.priors import GammaPrior
import pyro
import numpy as np
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt
import seaborn as sns

pyro.set_rng_seed(12345)
rng = np.random.default_rng(12345)


# Generate data for the simulations
n = 51
intercept = 1
slope = -1
noise_scale = 0.5**2

x = np.linspace(-3,3,n).reshape((n,1))
u = np.random.normal(size=(n,1))

simple_linear_lownoise = intercept*np.ones((n,1)) + slope*x - slope*x**2 + slope*x**3 # y1

# Normalize the training inputs
# train_x = (x - x.mean())/x.std()
train_x = torch.tensor(x.reshape((-1,)))

# Normalize the training outputs
train_y1 = (simple_linear_lownoise - simple_linear_lownoise.mean()) / simple_linear_lownoise.std() + noise_scale*u
train_y1 = torch.tensor(train_y1.reshape((-1,))) 


# Specify the model, likelihood, and parameters for prior distributions
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Specify the pyro_model
def pyro_model(x, y):
    with gpytorch.settings.fast_computations(False, False, False):
        sampled_model = model.pyro_sample_from_prior()
        output = sampled_model.likelihood(sampled_model(x))
        pyro.sample("obs", output, obs=y)
    return y

# Set up the sampler
NUM_SAMPLES = 200
NUM_WARMUP = 100
NUM_CHAINS = 1
kernel = NUTS(pyro_model)
mcmc_run = MCMC(kernel, 
                num_samples=NUM_SAMPLES, 
                warmup_steps=NUM_WARMUP, 
                num_chains = NUM_CHAINS,
                disable_progbar=False)

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = ExactGPModel(train_x, train_y1, likelihood)

model.covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(1, 2), "lengthscale")
model.covar_module.register_prior("outputscale_prior", GammaPrior(1, 5), "outputscale")
likelihood.register_prior("noise_prior", GammaPrior(1, 5), "noise")

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# See the error below this, if the error exists. No error means it finished training.
mcmc_run.run(train_x, train_y1)

mcmc_samples = mcmc_run.get_samples()
param_list = ['likelihood.noise_prior',  \
    'covar_module.outputscale_prior', 'covar_module.base_kernel.lengthscale_prior']
labels = ["noise", "os","ls"]
fig, axes = plt.subplots(nrows=2, ncols=2)
for i in range(3):
        samples = mcmc_samples[param_list[i]].numpy().reshape(-1)
        sns.scatterplot(range(NUM_SAMPLES) ,samples, ax=axes[int(i/2), int(i%2)])
        axes[int(i/2)][int(i%2)].legend([labels[i]])
plt.show()

model.pyro_load_from_samples(mcmc_run.get_samples())
model.eval()
test_x = torch.linspace(-3, 3, 101).unsqueeze(-1).double()
expanded_test_x = test_x.unsqueeze(0).repeat(NUM_SAMPLES, 1, 1).double()
output = model(expanded_test_x)
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y1.numpy(), 'k*', zorder=10)

    for i in range(min(100, 25)):
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), output.mean[i].detach().numpy(), 'b', linewidth=0.3)

    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Sampled Means'])
    plt.show()

