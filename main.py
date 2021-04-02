import torch
import json
import gpytorch
import pyro
import pickle
from pyro.infer.mcmc import NUTS, MCMC, HMC
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
num_samples = 2 if smoke_test else 200
warmup_steps = 2 if smoke_test else 100
# gpytorch.settings.cg_tolerance(1)


def train(train_x, train_i, train_y, model, likelihood, mll, optimizer):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x, train_i)
        loss = -mll(output, train_y.double())
        loss.backward()
        print('Iter %d/%d - LL: %.3f' % (i + 1, training_iterations, -loss.item()))
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
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)

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
    ohe = LabelEncoder()
    X = data.drop(columns=["station_id", "date", "national_politics", "sinclair2017",
    "post","affiliation","callsign"]).to_numpy().reshape(-1,) # , "weekday","affiliation","callsign"
    Group = data.sinclair2017.to_numpy().reshape(-1,1)
    ohe.fit(X)
    X = ohe.transform(X)
    station_le = LabelEncoder()
    ids = data.station_id.to_numpy().reshape(-1,)
    station_le.fit(ids)
    ids = station_le.transform(ids)
    # group/weekday/station effects
    X = np.concatenate((Group, X.reshape(-1,1),ids.reshape(-1,1),ds), axis=1)
    X_max_v = [np.max(X[:,i]).astype(int) for i in range(X.shape[1]-1)]
    Y = data.national_politics.to_numpy()
    T0 = date_le.transform(np.array([datetime.date(2017, 9, 1)]))
    train_condition = (data.post!=1) | (data.sinclair2017!=1)
    train_x = torch.Tensor(X[train_condition], device=device)
    train_y = torch.Tensor(Y[train_condition], device=device)

    idx = data.sinclair2017.to_numpy()
    train_i = torch.from_numpy(idx[train_condition]).to(device)

    test_x = torch.Tensor(X).double()
    test_y = torch.Tensor(Y).double()
    test_i = torch.from_numpy(idx)

    # fit = TwoWayFixedEffectModel(X_tr, X_co, Y_tr, Y_co, ATT, T0)
    # return
    
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
    # noise_prior = gpytorch.priors.UniformPrior(0.001,0.2)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior,\
        noise_constraint=gpytorch.constraints.Positive())
    model = MultitaskGPModel((train_x, train_i), train_y, X_max_v, likelihood)

    # fix some parameters
    # model.c_covar_module._set_c2(torch.var(train_y))
    model.x_covar_module[0].c2 = torch.var(train_y)
    for i in range(1,len(X_max_v)):
        model.x_covar_module[i].c2 = torch.tensor(0.05**2)
    # model.mean_module[0].constant.data.fill_(torch.mean(train_y[train_i==0]).double()) 
    # model.mean_module[1].constant.data.fill_(torch.mean(train_y[train_i==1]).double())
    # model.mean_module[0].constant.requires_grad = False
    # model.mean_module[1].constant.requires_grad = False
    # model.c_covar_module.raw_c2.requires_grad = False

    model.i_mean_module.bias.data.fill_(torch.mean(train_y[train_i==0]))
    slope = torch.mean(train_y[train_i==1])-torch.mean(train_y[train_i==0])
    model.i_mean_module.weights.data.fill_(slope)
    model.i_mean_module.weights.requires_grad = False
    model.i_mean_module.bias.requires_grad = False

    torch.set_default_tensor_type(torch.DoubleTensor)
    train_x, train_y, train_i = train_x.to(device), train_y.to(device), train_i.to(device)
    test_x, test_y, test_i = test_x.to(device), test_y.to(device), test_i.to(device)
    model.to(device)
    likelihood.to(device)

    # plot_prior(model)
    # return

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    def pyro_model(x, i, y):
        model.pyro_sample_from_prior()
        output = model(x, i)
        loss = mll.pyro_factor(output, y)
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

        # mlls = np.zeros(num_samples)
        # param_list = ["likelihood.noise_covar.noise_prior", "t_covar_module.outputscale_prior",
        #  "t_covar_module.base_kernel.lengthscale_prior", "task_covar_module.rho_prior"]
        # for i in range(500,800):
        #     model.task_covar_module._set_rho(np.min([1,mcmc_samples[param_list[3]].numpy().reshape(-1)[i]]))
        #     model.t_covar_module.outputscale = mcmc_samples[param_list[1]].numpy().reshape(-1)[i]**2 
        #     model.t_covar_module.base_kernel.lengthscale = mcmc_samples[param_list[2]].numpy().reshape(-1)[i]
        #     model.likelihood.noise_covar.noise = mcmc_samples[param_list[0]].numpy().reshape(-1)[i]**2
        #     output = model(train_x, train_i)
        #     mlls[i] = mll(output, train_y.double())
        # import seaborn as sns
        # from matplotlib import pyplot as plt
        # sns.distplot(mlls[500:800])
        # plt.show()
        # return
        plot_posterior(mcmc_samples)
        return
        for k, d in mcmc_samples.items():
            mcmc_samples[k] = d[idx]
        model.pyro_load_from_samples(mcmc_samples)
        visualize_localnews_MCMC(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
                likelihood, T0, date_le, station_le, 10)
        return
        
    elif INFERENCE=='MAP':
        model.task_covar_module._set_rho(0.9)
        model.t_covar_module.outputscale = 0.02**2 
        model.t_covar_module.base_kernel.lengthscale = 3
        model.likelihood.noise_covar.noise = 0.035**2

        # k = 50
        # rhos = np.linspace(-0.99,0.99,k)
        # oss = np.linspace(0.001,0.499,k)

        # model.t_covar_module.base_kernel.lengthscale = 7
        # model.likelihood.noise_covar.noise = 0.035**2
        # losses = np.zeros((k,k))
        # for i in range(k):
        #     for j in range(k):
        #         print(i*k+j)
        #         model.task_covar_module._set_rho(rhos[i])
        #         model.t_covar_module.outputscale = oss[j]**2
        #         model.train()
        #         likelihood.train()
        #         output = model(train_x, train_i)
        #         loss = mll(output, train_y.double())*train_x.shape[0]
        #         losses[i,j] = loss

        # import seaborn as sns
        # from matplotlib import pyplot as plt
        # sns.set(rc={'figure.figsize':(20,16)})
        # ax = sns.heatmap(losses, cmap="YlGnBu")
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=20)

        # np.savetxt("results/heatmapls7.csv", losses, delimiter=",")
        # plt.savefig("./results/heatmapls7.pdf")
        # plt.close()
        # return
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model, likelihood = train(train_x, train_i, train_y, model, likelihood, mll, optimizer)
        torch.save(model.state_dict(), 'results/localnews_' +  INFERENCE + '_model_state.pth')
        visualize_localnews(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
         likelihood, T0, date_le, station_le)
        return
    elif INFERENCE=='MCMC':
        model.task_covar_module._set_rho(0.9)
        model.t_covar_module.outputscale = 0.02**2 
        model.t_covar_module.base_kernel.lengthscale = 3
        model.likelihood.noise_covar.noise = 0.035**2
        initial_params = {"likelihood.noise_covar.noise_prior": model.likelihood.raw_noise.detach(), 
         "t_covar_module.outputscale_prior": model.t_covar_module.raw_outputscale.detach(),
         "t_covar_module.base_kernel.lengthscale_prior": model.t_covar_module.base_kernel.raw_lengthscale.detach(),
         "task_covar_module.rho_prior": model.task_covar_module.raw_rho.detach()}

        nuts_kernel = NUTS(pyro_model, adapt_step_size=True, adapt_mass_matrix=True)
        hmc_kernel = HMC(pyro_model, step_size=1e-1, num_steps=4, adapt_step_size=True)
        mcmc_run = MCMC(hmc_kernel, num_samples=num_samples, warmup_steps=warmup_steps, initial_params=initial_params)
        mcmc_run.run(train_x, train_i, train_y)
        # save the posterior
        # with open('results/localnews_' + INFERENCE+ '.pkl', 'wb') as f:
        #     pickle.dump(mcmc_run.get_samples(), f)
        pickle.dump(mcmc_run, open("results/localnews_MCMC.pkl", "wb"))
        torch.save(model.state_dict(), 'results/localnews_' + INFERENCE +'_model_state.pth')
        return
        visualize_localnews_MCMC(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
                likelihood, T0, date_le, station_le, num_samples)
    else:
        model.load_strict_shapes(False)
        state_dict = torch.load('results/localnews_MAP_model_state.pth')
        model.load_state_dict(state_dict)
        print(f'Parameter name: rho value = {model.task_covar_module.rho.detach().numpy()}')
        print(f'Parameter name: ls value = {model.t_covar_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: os value = {np.sqrt(model.t_covar_module.outputscale.detach().numpy())}')
        print(f'Parameter name: noise value = {np.sqrt(model.likelihood.noise.detach().numpy())}')
        # for param_name, param in model.named_parameters():
        #     print(f'Parameter name: {param_name:42} value = {param.detach().numpy()}')
        visualize_localnews(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
         likelihood, T0, date_le, station_le)

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
