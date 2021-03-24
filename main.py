import torch
import json
import gpytorch
import pyro
import pickle
from pyro.infer.mcmc import NUTS, MCMC
from model.multitaskmodel import MultitaskGPModel
from utilities.savejson import savejson
from utilities.visualize import visualize_synthetic, plot_posterior
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


smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 200
num_samples = 2 if smoke_test else 1
warmup_steps = 2 if smoke_test else 2


def train(train_x, train_i, train_y, model, likelihood, mll, optimizer):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x, train_i)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        # print(f'Parameter name: task_covar_module.rho value = {model.task_covar_module.rho.detach().numpy()}')
        # print(f'Parameter name: task_covar_module.raw_rho value = {model.task_covar_module.raw_rho.detach().numpy()}')
        optimizer.step()

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
    # train_i_tr = [torch.full_like(Y_tr[i,:T0].reshape(-1), dtype=torch.long, fill_value=i)
    #         for i in range(X_tr.shape[0])] 
    # train_i_co = [torch.full_like(Y_co[i].reshape(-1), dtype=torch.long, fill_value=i+N_tr)
    #         for i in range(X_co.shape[0])]

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

    # initialize parameter
    # model.task_covar_module.covar_factor = torch.nn.Parameter(torch.zeros(1,2))
    
    B = model.task_covar_module.covar_factor
    v = model.task_covar_module.raw_var
    print(torch.matmul(B,B.T)+torch.exp(v))

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

    B = model.task_covar_module.covar_factor
    v = model.task_covar_module.raw_var
    print(torch.matmul(B,B.T)+torch.exp(v))

    # json_file = "results/optimizedhyps.json"
    # savejson(model, likelihood, json_file)

    visualize_synthetic(X_tr, X_co, Y_tr, Y_co, ATT, model, likelihood, T0)


def localnews(INFERENCE):
    with open('model/conf.json') as f:
        configs = json.load(f)
    sigma_noise = configs["sigma_noise"]

    data = pd.read_csv("data/localnews.csv",index_col=[0])
    # data.national_politics = np.log(data.national_politics/(1-data.national_politics))
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())
    # data = data.sort_values(by=['date'])
    # data = data[(data.date<=datetime.date(2017, 9, 10)) & (data.date>=datetime.date(2017, 8, 20))]
    # data = data[data.station_id.isin([1345,1350])]
    N = data.station_id.unique().shape[0]

    date_le = LabelEncoder()
    ds = data.date
    date_le.fit(ds)
    ds = date_le.transform(ds).reshape((-1,1))
    ohe = OneHotEncoder() 
    X = data.drop(columns=["station_id", "date", "national_politics", 
    "sinclair2017", "post","weekday","affiliation","callsign"]) #  "weekday","affiliation","callsign"
    ohe.fit(X)
    X = ohe.transform(X).toarray()
    station_le = LabelEncoder()
    ids = data.station_id
    station_le.fit(ids)
    ids = station_le.transform(ids)
    X = np.concatenate((X,ds), axis=1)
    Y = data.national_politics.to_numpy()
    T0 = date_le.transform(np.array([datetime.date(2017, 9, 1)]))
    
    train_condition = (data.post!=1) | (data.sinclair2017!=1)
    train_x = torch.Tensor(X[train_condition]).double()
    train_y = torch.Tensor(Y[train_condition]).double()

    idx = data.sinclair2017.to_numpy()
    train_i = torch.from_numpy(idx[train_condition])

    test_x = torch.Tensor(X).double()
    test_y = torch.Tensor(Y).double()
    test_i = torch.from_numpy(idx)
    # fit = TwoWayFixedEffectModel(X_tr, X_co, Y_tr, Y_co, ATT, T0)
    # return
    
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior,\
        noise_constraint=gpytorch.constraints.Positive())
    model = MultitaskGPModel((train_x, train_i), train_y, N, likelihood)

    # fix some parameters
    model.c_covar_module._set_c2(torch.var(train_y))
    # model.mean_module[0].constant.data.fill_(torch.mean(train_y[train_i==0]).double()) 
    # model.mean_module[1].constant.data.fill_(torch.mean(train_y[train_i==1]).double())
    # model.mean_module[0].constant.requires_grad = False
    # model.mean_module[1].constant.requires_grad = False
    model.c_covar_module.raw_c2.requires_grad = False

    model.i_mean_module.bias.data.fill_(torch.mean(train_y[train_i==0]).double())
    slope = torch.mean(train_y[train_i==1]).double()-torch.mean(train_y[train_i==0]).double()
    model.i_mean_module.weights.data.fill_(slope)
    model.i_mean_module.weights.requires_grad = False
    model.i_mean_module.bias.requires_grad = False

    model.double()
    likelihood.double()

    # plot_prior(model)
    # return

    # visualize_localnews(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
    #      likelihood, T0, date_le, station_le)
    # return

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, i, y):
        model.pyro_sample_from_prior()
        output = model(x, i)
        loss = mll.pyro_factor(output, y)
        return y

    # B = model.task_covar_module.base_kernel.covar_factor
    # v = model.task_covar_module.base_kernel.raw_var
    # print(torch.matmul(B,B.T)+torch.exp(v))

    if not os.path.isdir("results"):
        os.mkdir("results")


    if INFERENCE=='MCMCLOAD':
        # model.load_strict_shapes(False)
        # state_dict = torch.load('results/localnews_MCMC_model_state.pth')
        # model.load_state_dict(state_dict)
        # for param_name, param in model.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param.detach().numpy()}')
        with open('results/localnews_MCMC.pkl', 'rb') as f:
            mcmc_run = pickle.load(f)
        mcmc_samples = mcmc_run.get_samples()
        model.pyro_load_from_samples(mcmc_samples)
        plot_posterior(mcmc_samples)
        visualize_localnews_MCMC(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
                likelihood, T0, date_le, station_le, num_samples)
        
    elif INFERENCE=='MAP':
        model.task_covar_module._set_rho(0.0)
        model.t_covar_module.outputscale = 0.05**2 
        model.t_covar_module.base_kernel.lengthscale = 14
        model.likelihood.noise_covar.noise = 0.05**2

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        model, likelihood = train(train_x, train_i, train_y, model, likelihood, mll, optimizer)
        torch.save(model.state_dict(), 'results/localnews_' +  INFERENCE + '_model_state.pth')
    elif INFERENCE=='MCMC':
        nuts_kernel = NUTS(pyro_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=smoke_test)
        mcmc_run.run(train_x, train_i, train_y)
        # save the posterior
        # with open('results/localnews_' + INFERENCE+ '.pkl', 'wb') as f:
        #     pickle.dump(mcmc_run.get_samples(), f)
        pickle.dump(mcmc_run, open("results/localnews_MCMC.pkl", "wb"))
        torch.save(model.state_dict(), 'results/localnews_' + INFERENCE +'_model_state.pth')
        
    else:
        model.load_strict_shapes(False)
        state_dict = torch.load('results/localnews_MAP_model_state.pth')
        model.load_state_dict(state_dict)
        for param_name, param in model.named_parameters():
            print(f'Parameter name: {param_name:42} value = {param.detach().numpy()}')

    visualize_localnews(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
         likelihood, T0, date_le, station_le)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python main.py --type localnews')
    parser.add_argument('-t','--type', help='localnews/synthetic', required=True)
    parser.add_argument('-i','--inference', help='MCMC/MAP/MAPLOAD/MCMCLOAD', required=True)
    args = vars(parser.parse_args())
    if args['type'] == 'localnews':
        localnews(INFERENCE=args['inference'])
    elif args['type'] == 'synthetic': 
        synthetic(INFERENCE=args['inference'])
    else:
        exit()
