import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.lazy import InterpolatedLazyTensor
from gpytorch.utils.broadcasting import _mul_broadcast_shape
from gpytorch.constraints import Positive
import torch
import time

from gpytorch.kernels.kernel import Kernel

num_samples = 100
warmup_steps = 100

torch.set_default_tensor_type(torch.DoubleTensor)

class myIndexKernel(Kernel):
    r"""
    A kernel for discrete indices. Kernel is defined by a lookup table.

    .. math::

        \begin{equation}
            k(i, j) = v_i if i==j otherwise 0
        \end{equation}

    where :math:`\mathbf v` is a  non-negative vector.
    These parameters are learned.

    Args:
        :attr:`num_tasks` (int):
            Total number of indices.
        :attr:`batch_shape` (torch.Size, optional):
        :attr:`prior` (:obj:`gpytorch.priors.Prior`):
            Prior for :math:`v` vector.

    Attributes:
        raw_var:
            The element-wise log of the :math:`\mathbf v` vector.
    """

    def __init__(self, num_tasks, prior=None, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks
        self.register_parameter(name="raw_var", parameter=torch.nn.Parameter(torch.randn(*self.batch_shape, 1)))
        if prior is not None:
            self.register_prior("raw_var_prior", prior, lambda m: m.var,
             lambda m, v: m._set_var(v))

        self.register_constraint("raw_var", Positive())

    @property
    def var(self):
        return self.raw_var_constraint.transform(self.raw_var)

    @var.setter
    def var(self, value):
        self._set_var(value)

    def _set_var(self, value):
        self.initialize(raw_var=self.raw_var_constraint.inverse_transform(value))

    def _eval_covar_matrix(self):
        return torch.diag_embed(self.var*torch.ones((self.num_tasks,)))

    def forward(self, i1, i2, **params):

        i1, i2 = i1.long(), i2.long()
        covar_matrix = self._eval_covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], i2.shape[:-2], self.batch_shape)

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.expand(batch_shape + i1.shape[-2:]),
            right_interp_indices=i2.expand(batch_shape + i2.shape[-2:]),
        )
        return res

class FEGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_task):
        super(FEGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.x_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.i_covar_module = myIndexKernel(num_tasks=num_task, prior=GammaPrior(1,1))
    def forward(self, x):
        mean_x = self.mean_module(x[:,0])
        covar_x = self.x_covar_module(x[:,0])
        covar_i = self.i_covar_module(x[:,1])
        covar = covar_x + covar_i 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

def main():
    # simulate synthetic data
    pyro.set_rng_seed(12345)
    rng = np.random.default_rng(12345)
    n = 500
    T = 10
    noise_scale = 0.5**2

    # TODO: 500 units grouped by 10 clusters

    # fixed effects
    x = np.arange(0,T)
    fe_scale = 2
    fix_effects = np.random.normal(loc=0,scale=fe_scale, size=(n,))
    train_x = np.zeros((n*T,2))
    train_y = np.zeros((n*T,))
    # x=[t, i], y=t/10+2*sin(t/5)+N(0,1)
    for i in range(n):
        for t in range(T):
            train_x[i*T+t,0] = t
            train_x[i*T+t,1] = i
            train_y[i*T+t] = 0.1*t + 2*np.sin(t/5)\
                 + noise_scale*np.random.normal(size=(1,)) + fix_effects[i]
    
    train_x = torch.tensor(train_x)
    train_y = torch.tensor(train_y.reshape((-1,)))

    # Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=Positive(),
            noise_prior=GammaPrior(1,2))
    model = FEGPModel(train_x, train_y, likelihood, num_task=n)

    model.x_covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(1, 1/10), "lengthscale")
    model.x_covar_module.register_prior("outputscale_prior", GammaPrior(1, 1/2), "outputscale")
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0),
        'x_covar_module.base_kernel.lengthscale': torch.tensor(10.0),
        'x_covar_module.outputscale': torch.tensor(1.0),
        'i_covar_module.var': torch.tensor(1.0)
    }

    model.initialize(**hypers)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    for i in range(0):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    # Initialize plots
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
    test_x = train_x
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_y1 = likelihood(model(train_x[0:T]))
        observed_pred_y2 = likelihood(model(train_x[T:(2*T)]))

    # Define plotting function
    def ax_plot(ax, train_y, train_x, rand_var, title):
        # Get lower and upper confidence bounds
        m = rand_var.mean
        v = rand_var.covariance_matrix.diag()
        lower, upper = m-2*v, m+2*v
        # Plot training data as black stars
        ax.plot(train_x[:,0].detach().numpy(), train_y.detach().numpy(), 'k*')
        # Predictive mean as blue line
        ax.plot(test_x[0:T,0].detach().numpy(), rand_var.mean.detach().numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x[0:T,0].detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        ax.set_ylim([-5, 5])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.set_title(title)

    # plot MAP
    ax_plot(y1_ax, train_y[0:T], train_x[0:T], observed_pred_y1, 'Observed Values (Likelihood)')
    ax_plot(y2_ax, train_y[T:(2*T)], train_x[T:(2*T)], observed_pred_y2, 'Observed Values (Likelihood)')
    # plt.show()

    # define pyro model
    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
            loss = -mll(output,y)
        return loss

    model.double()
    likelihood.double()

    # Sample model hyperparameters
    model.train()
    likelihood.train()

    nuts_kernel = NUTS(pyro_model, adapt_step_size=True, jit_compile=False)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    return model, likelihood, mll, mcmc_run, train_x, train_y

if __name__ == "__main__":
    start = time.time()
    model, likelihood, mll, mcmc_run, train_x, train_y = main()
    mcmc_run.run(train_x, train_y)

    end = time.time()
    print(end - start)
    mcmc_samples = mcmc_run.get_samples()
    # plot sampler trace
    param_list = ['likelihood.noise_covar.noise_prior',  \
        'x_covar_module.outputscale_prior', 'x_covar_module.base_kernel.lengthscale_prior',\
            'i_covar_module.raw_var_prior']
    labels = ["noise", "os","ls", "v"]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    for i in range(4):
         samples = mcmc_samples[param_list[i]].numpy().reshape(-1)
         sns.scatterplot(range(num_samples) ,samples, ax=axes[int(i/2), int(i%2)])
         axes[int(i/2)][int(i%2)].legend([labels[i]])
    plt.show()


    # model.pyro_load_from_samples(mcmc_run.get_samples())
    # model.eval()
    # test_x = torch.linspace(-2, 2, 101).unsqueeze(-1)
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
    #     plt.show()

