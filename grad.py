import math
import torch
import gpytorch
from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
import pyro
import copy
from pyro.infer.mcmc import NUTS, MCMC, HMC

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

x = torch.linspace(0, 1, 10).double()
y = torch.sin(x * (2 * math.pi)).double() + torch.randn(x.size()).double() * 0.1

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.double()
likelihood.register_prior("raw_noise_prior", NormalPrior(0, 1), "raw_noise")

model = ExactGPModel(x, y, likelihood)
model.double()
model.mean_module.register_prior("mean_prior", NormalPrior(0, 1), "constant")
model.covar_module.base_kernel.register_prior("raw_lengthscale_prior", NormalPrior(0, 1), "raw_lengthscale")
model.covar_module.register_prior("raw_outputscale_prior", NormalPrior(0, 1), "raw_outputscale")

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# define pyro primitive
def pyro_model(x, y):
    model.pyro_sample_from_prior()
    output = model(x)
    with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        loss = mll(output, y) * y.shape[0]
    pyro.factor("gp_mll", loss)

# not sure where these values came from
initial_params = \
       {'mean_module.mean_prior':                         torch.DoubleTensor([0.5]),\
        'covar_module.base_kernel.raw_lengthscale_prior': torch.DoubleTensor([0.5413]),\
        'covar_module.raw_outputscale_prior':             torch.DoubleTensor([-1.2587]),\
        'likelihood.raw_noise_prior':                     torch.DoubleTensor([-1.2587])}

args = (x, y)
kwargs = {}
nuts_kernel = NUTS(pyro_model)
nuts_kernel.setup(0, *args, **kwargs)


# check gradient
d = 1e-6
grads, v = pyro.ops.integrator.potential_grad(nuts_kernel.potential_fn, initial_params)

print(v)
print(initial_params)

for param in initial_params:
    new_params = copy.deepcopy(initial_params)
    new_params[param] = new_params[param] + d
    v2 = nuts_kernel.potential_fn(new_params).item()
    numerical = (v2 - v) / d
    # print(f"pyro thinks the gradient for {param} is {grads[param]}, numerically it's {numerical}")