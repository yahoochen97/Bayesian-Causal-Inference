import torch
import json
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from model.multitaskmodel import MultitaskGPModel
from utilities.savejson import savejson
from utilities.visualize import visualize_synthetic, plot_posterior, plot_pyro_posterior
from utilities.visualize import visualize_localnews, visualize_localnews_MCMC, plot_prior
from utilities.synthetic import generate_synthetic_data
from model.fixedeffect import TwoWayFixedEffectModel
import os
import pandas as pd
import numpy as np
import argparse
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import dill as pickle
import sampyl as smp


smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 15
num_samples = 2 if smoke_test else 500
warmup_steps = 2 if smoke_test else 500


def train(train_x, train_y, model, likelihood, mll, optimizer):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    current_ll = 0
    current_state = model.state_dict()
    for i in range(training_iterations):
        
        # optimizer.zero_grad()
        # output = model(train_x)
        # with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        #     loss = -mll(output, train_y)*train_y.shape[0]
        # loss.backward()
        # if -loss.item() > current_ll:
        #     current_ll = -loss.item()
        #     current_state = model.state_dict()

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

        # print('Iter %d/%d - LL: %.3f' % (i + 1, training_iterations, -loss.item()))
        optimizer.step(closure)

    # model.load_state_dict(current_state)
    # print(current_ll)

    return model, likelihood


def synthetic(INFERENCE):
    # load configurations
    with open('model/conf.json') as f:
        configs = json.load(f)

    N_tr = configs["N_tr"]
    N_co = configs["N_co"]
    N = N_tr + N_co
    T = configs["T"]
    T0 = configs["T0"]
    d = configs["d"]
    noise_std = configs["noise_std"]
    Delta = configs["treatment_effect"]
    seed = configs["seed"]

    X_tr, X_co, Y_tr, Y_co, ATT = generate_synthetic_data(N_tr, N_co, T, T0, d, Delta, noise_std, seed)
    train_x_tr = X_tr[:,:T0].reshape(-1,d+1)
    train_x_co = X_co.reshape(-1,d+1)
    train_y_tr = Y_tr[:,:T0].reshape(-1)
    train_y_co = Y_co.reshape(-1)

    train_x = torch.cat([train_x_tr, train_x_co])
    train_y = torch.cat([train_y_tr, train_y_co])

    # treat group 1, control group 0
    train_i_tr = torch.full_like(train_y_tr, dtype=torch.long, fill_value=1)
    train_i_co = torch.full_like(train_y_co, dtype=torch.long, fill_value=0)
    train_i = torch.cat([train_i_tr, train_i_co])
        
    # fit = TwoWayFixedEffectModel(X_tr, X_co, Y_tr, Y_co, ATT, T0)
    # return 
    # train_x, train_y, train_i = build_gpytorch_data(X_tr, X_co, Y_tr, Y_co, T0)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultitaskGPModel((train_x, train_i), train_y, N, likelihood)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, i, y):
        model.pyro_sample_from_prior()
        output = model(x, i)
        loss = mll.pyro_factor(output, y)
        return y

    if not os.path.isdir("results"):
        os.mkdir("results")

    if INFERENCE=='MAPLOAD':
        model.load_strict_shapes(False)
        state_dict = torch.load('results/synthetic_MAP_model_state.pth')
        model.load_state_dict(state_dict)
    elif INFERENCE=="MAP":
         # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        model, likelihood = train(train_x, train_i, train_y, model, likelihood, mll, optimizer)
        torch.save(model.state_dict(), 'results/synthetic_' + INFERENCE + '_model_state.pth')
    else:
        nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=smoke_test)
        mcmc_run.run(train_x, train_i, train_y)
        torch.save(model.state_dict(), 'results/synthetic_' + INFERENCE +'_model_state.pth')

    visualize_synthetic(X_tr, X_co, Y_tr, Y_co, ATT, model, likelihood, T0)


def localnews(INFERENCE):
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    # preprocess data
    data = pd.read_csv("data/localnews.csv",index_col=[0])
    N = data.station_id.unique().shape[0]
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
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
    train_x = torch.Tensor(X[train_condition], device=device).double()
    train_y = torch.Tensor(Y[train_condition], device=device).double()

    idx = data.sinclair2017.to_numpy()
    train_g = torch.from_numpy(idx[train_condition]).to(device)

    test_x = torch.Tensor(X).double()
    test_y = torch.Tensor(Y).double()
    test_g = torch.from_numpy(idx)
    
    # define likelihood
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior,\
        noise_constraint=gpytorch.constraints.Positive())
    model = MultitaskGPModel(train_x, train_y, X_max_v, likelihood)

    # group effects
    # model.x_covar_module[0].c2 = torch.var(train_y)
    # model.x_covar_module[0].raw_c2.requires_grad = False

    # weekday/day/unit effects initialize to 0.05**2
    for i in range(len(X_max_v)):
        model.x_covar_module[i].c2 = torch.tensor(0.05**2)

    # fix unit mean/variance by not requiring grad
    model.x_covar_module[-1].raw_c2.requires_grad = False
    model.unit_mean_module.constant.data.fill_(0.12)
    model.unit_mean_module.constant.requires_grad = False

    # set precision to double tensors
    torch.set_default_tensor_type(torch.DoubleTensor)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    model.to(device)
    likelihood.to(device)

    # define Loss for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, y):
        model.pyro_sample_from_prior()
        output = model(x)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            loss = mll(output, y)*y.shape[0]
        pyro.factor("gp_mll", loss)
        return y

    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_i = train_i.cuda()
        train_y = train_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

    if not os.path.isdir("results"):
        os.mkdir("results")

    if INFERENCE=='MCMCLOAD':
        plot_prior(model)
        with open('results/localnews_MCMC.pkl', 'rb') as f:
            mcmc_run = pickle.load(f)
        mcmc_samples = mcmc_run.get_samples()
        print(mcmc_run.summary())
        return
        plot_pyro_posterior(mcmc_samples)
        # plot_posterior(mcmc_samples)
        return
        for k, d in mcmc_samples.items():
            mcmc_samples[k] = d[idx]
        model.pyro_load_from_samples(mcmc_samples)
        visualize_localnews_MCMC(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
                likelihood, T0, station_le, 10)
        return
        
    elif INFERENCE=='MAP':
        model.group_index_module._set_rho(0.0)
        model.group_t_covar_module.outputscale = 0.05**2  
        model.group_t_covar_module.base_kernel.lengthscale = 15
        model.likelihood.noise_covar.noise = 0.05**2
        model.unit_t_covar_module.outputscale = 0.05**2 
        model.unit_t_covar_module.base_kernel.lengthscale = 40
        # weekday/day/unit effects initialize to 0.05**2
        for i in range(len(X_max_v)):
            model.x_covar_module[i].c2 = torch.tensor(0.05**2)
        
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer = torch.optim.LBFGS(model.parameters())
        model, likelihood = train(train_x, train_y, model, likelihood, mll, optimizer)
        torch.save(model.state_dict(), 'results/localnews_' +  INFERENCE + '_model_state.pth')
        visualize_localnews(data, test_x, test_y, test_g, model, likelihood, T0, station_le)
        return
    elif INFERENCE=='MCMC':
        # model.group_index_module._set_rho(0.5)
        # model.group_t_covar_module.outputscale = 0.05**2 
        # model.group_t_covar_module.base_kernel.lengthscale = 15
        # model.likelihood.noise_covar.noise = 0.05**2
        # model.unit_t_covar_module.outputscale = 0.05**2 
        # model.unit_t_covar_module.base_kernel.lengthscale = 40
       
        # weekday/day/unit effects initialize to 0.05**2

        for i in range(len(X_max_v)-1):
            model.x_covar_module[i].c2 = torch.tensor(0.0**2)
            model.x_covar_module[i].raw_c2.requires_grad = False

        # initial_params =  {'task_covar_module.rho_prior': model.task_covar_module.raw_rho.detach(),\
        #     't_covar_module.base_kernel.lengthscale_prior':  model.t_covar_module.base_kernel.raw_lengthscale.detach(),\
        #     't_covar_module.outputscale_prior': model.t_covar_module.raw_outputscale.detach(),\
        #     'unit_t_covar_module.base_kernel.lengthscale_prior':  model.unit_t_covar_module.base_kernel.raw_lengthscale.detach(),\
        #     'unit_t_covar_module.outputscale_prior': model.unit_t_covar_module.raw_outputscale.detach(),\
        #     'likelihood.noise_covar.noise_prior': model.likelihood.raw_noise.detach()
        # }

        nuts_kernel = NUTS(pyro_model, adapt_step_size=True, adapt_mass_matrix=True)
        hmc_kernel = HMC(pyro_model, step_size=0.1, num_steps=5, adapt_step_size=True)
        mcmc_run = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps)#, initial_params=initial_params)
        mcmc_run.run(train_x, train_y)
        pickle.dump(mcmc_run, open("results/localnews_MCMC.pkl", "wb"))
        plot_pyro_posterior(mcmc_run.get_samples())

        return

        visualize_localnews_MCMC(data, train_x, train_y, train_g, test_x, test_y, test_i, model,\
                likelihood, T0,  station_le, num_samples)
    else:
        model.load_strict_shapes(False)
        state_dict = torch.load('results/localnews_MAP_model_state.pth')
        model.load_state_dict(state_dict)
        output = model(train_x)
        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            loss = mll(output, train_y)*train_y.shape[0]
        print(f'LL: = {loss}')
        print(f'Parameter name: rho value = {model.group_index_module.rho.detach().numpy()}')
        print(f'Parameter name: group ls value = {model.group_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: group os value = {np.sqrt(model.group_t_covar_module.outputscale.detach().numpy())}')
        print(f'Parameter name: unit ls value = {model.unit_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: unit os value = {np.sqrt(model.unit_t_covar_module.outputscale.detach().numpy())}')
        print(f'Parameter name: noise value = {np.sqrt(model.likelihood.noise.detach().numpy())}')
        print(f'Parameter name: weekday std value = {np.sqrt(model.x_covar_module[0].c2.detach().numpy())}')
        print(f'Parameter name: day std value = {np.sqrt(model.x_covar_module[1].c2.detach().numpy())}')
        visualize_localnews(data, test_x, test_y, test_g, model, likelihood, T0, station_le)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python main.py --type localnews --inference MAP')
    parser.add_argument('-t','--type', help='localnews/synthetic', required=True)
    parser.add_argument('-i','--inference', help='MCMC/MAP/MAPLOAD/MCMCLOAD', required=True)
    args = vars(parser.parse_args())
    if args['type'] == 'localnews':
        localnews(INFERENCE=args['inference'])
    elif args['type'] == 'synthetic': 
        synthetic(INFERENCE=args['inference'])
    else:
        exit()
