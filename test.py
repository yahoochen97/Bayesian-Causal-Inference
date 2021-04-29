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

import seaborn as sns

import numpy as np
from model.multitaskmodel import PresentationModel
import pandas as pd
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

    train_x = torch.linspace(0, 1, 1000).double()
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


def presentation():
    INFERENCE = 'MAP'
    torch.set_default_tensor_type(torch.DoubleTensor)
    # preprocess data
    data = pd.read_csv("data/localnews.csv",index_col=[0])
    N = data.station_id.unique().shape[0]
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())

    # data = data[(data.date<=datetime.date(2017, 9, 10)) & (data.date>=datetime.date(2017, 8, 20))]
    
    ds = data.t.to_numpy().reshape((-1,1))
    ohe = OneHotEncoder()
    ohe = LabelEncoder()
    X = data.drop(columns=["station_id", "date", "national_politics", "sinclair2017",
    "post","affiliation","callsign","t"]).to_numpy().reshape(-1,) # , "weekday","affiliation","callsign"
    Group = data.sinclair2017.to_numpy().reshape(-1,1)
    ohe.fit(X)
    X = ohe.transform(X)
    station_le = LabelEncoder()
    ids = data.station_id.to_numpy().reshape(-1,)
    station_le.fit(ids)
    ids = station_le.transform(ids)
    # weekday/day/unit effects and time trend
    X = np.concatenate((X.reshape(-1,1),ds,ids.reshape(-1,1),Group,ds), axis=1)
    # numbers of dummies for each effect
    X_max_v = [np.max(X[:,i]).astype(int) for i in range(X.shape[1]-2)]

    Y = data.national_politics.to_numpy()
    T0 = data[data.date==datetime.date(2017, 9, 1)].t.to_numpy()[0]
    train_condition = (data.post!=1) | (data.sinclair2017!=1)
    train_x = torch.Tensor(X[train_condition]).double()
    train_y = torch.Tensor(Y[train_condition]).double()

    idx = data.sinclair2017.to_numpy()
    train_g = torch.from_numpy(idx[train_condition])

    test_x = torch.Tensor(X).double()
    test_y = torch.Tensor(Y).double()
    test_g = torch.from_numpy(idx)
    
    # define likelihood
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior if "MAP" in INFERENCE else None,\
            noise_constraint=gpytorch.constraints.Positive())

    model = PresentationModel(train_x, train_y, X_max_v, likelihood, MAP="MAP" in INFERENCE)

    # for i in range(len(X_max_v)):
    #     model.x_covar_module[i].c2 = torch.tensor(0.05**2)

    # fix unit mean/variance by not requiring grad
    # model.x_covar_module[-1].raw_c2.requires_grad = False

    # model.unit_mean_module.constantvector.data.fill_(torch.tensor([0.12,0.12]))
    # model.unit_mean_module.constant.requires_grad = True
    model.group_mean_module.constantvector.data[0].fill_(0.11)
    model.group_mean_module.constantvector.data[1].fill_(0.12)

    rho = 0.9
    model.group_index_module._set_rho(rho)
    model.group_t_covar_module.outputscale = 0.02**2  
    model.group_t_covar_module.base_kernel.lengthscale = 7
    model.likelihood.noise_covar.noise = 0.03**2
    model.unit_t_covar_module.outputscale = 0.016**2 
    model.unit_t_covar_module.base_kernel.lengthscale = 30

    # weekday/day/unit effects initialize to 0.01**2
    # for i in range(len(X_max_v)):
    #     model.x_covar_module[i].c2 = torch.tensor(0.05**2)
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, history_size=10, max_iter=4)

    model.train()
    likelihood.train()
    training_iterations = 0
    for i in range(training_iterations):
    
        def closure():
            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(train_x)
            # Compute loss
            with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                loss = -mll(output, train_y)*train_x.shape[0]
            print('Iter %d/%d - LL: %.3f' % (i + 1, training_iterations, -loss.item()))
            # Backward pass
            loss.backward()
            return loss

        optimizer.step(closure=closure)

    # print(f'Parameter name: rho value = {model.group_index_module.rho.detach().numpy()}')
    # print(f'Parameter name: group ls value = {model.group_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
    # print(f'Parameter name: group os value = {np.sqrt(model.group_t_covar_module.outputscale.detach().numpy())}')
    # print(f'Parameter name: unit ls value = {model.unit_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
    # print(f'Parameter name: unit os value = {np.sqrt(model.unit_t_covar_module.outputscale.detach().numpy())}')
    # print(f'Parameter name: noise value = {np.sqrt(model.likelihood.noise.detach().numpy())}')
    # print(model.unit_mean_module.constantvector.data)
    # print(model.group_mean_module.constantvector.data)


    model.unit_t_covar_module.outputscale = 0

    model.eval()
    likelihood.eval()

     # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
         f_pred = model(test_x)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()
    
    station_ids = data.station_id.unique()
    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "upper": upper,
         "lower": lower,
         "m": f_pred.mean,
         "y": test_y,
         "sinclair2017": data.sinclair2017})
    result = result.groupby(['t','g'], as_index=False)[['lower','upper','m','y']].mean()
    fill_alpha = [0.2, 0.5]
    mean_color = ["blue", "red"]
    y_color = ["blue", "red"]
    plt.rcParams["figure.figsize"] = (10,6)
    for g in [0,1]:
         test_t = np.unique(result[result.g==g].t)
         lower_g = result[result.g==g].lower.to_numpy()
         upper_g = result[result.g==g].upper.to_numpy()
         m_g = result[result.g==g].m.to_numpy()
         if g==0:
              m_g_0 = m_g
         else:
              m_g_1 = m_g
         y_g = result[result.g==g].y.to_numpy()
         LABEL = "Acquired" if g==1 else "Not Acquired"

         plt.scatter(x=1+test_t, y=y_g, c=y_color[g], s=4, label=LABEL + " data")
         plt.plot(1+test_t, m_g, c=mean_color[g], linewidth=2, label=LABEL +' estimated Y(0)')
         # plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[g], label=LABEL + " 95% CI")
         plt.legend(loc=2)
         plt.title("Averaged Group Trends Correlation {}".format(rho))
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
    plt.savefig("results/presentation/localnews_MAP.png")
    plt.show()


def drift():


if __name__ == "__main__":
    presentation()
    exit()

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

