import torch
import gpytorch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.special as sps 
import seaborn as sns
import copy


# Define plotting function
def ax_plot(ax, test_t, X, Y, m, lower, upper, LABEL):
     for i in range(2):
          ax[i].plot(1+X[i, :,-1].detach().numpy(), Y[i,:].detach().numpy(),\
          color='grey', alpha=0.8, label=LABEL)
          ax[i].plot(1+test_t.detach().numpy(), m[i].detach().numpy(),\
               'k--', linewidth=1.0, label='Estimated Y(0)')
          ax[i].fill_between(1+test_t.detach().numpy(), lower[i].detach().numpy(),\
               upper[i].detach().numpy(), alpha=0.5)
          ax[i].legend(loc=2)
          ax[i].set_title("{} Unit {}".format(LABEL, i+1))


def plot_pyro_prior(priors, transforms):
    labels = ["rho", "group ls", "group os", "unit ls", "unit os", "noise","weekday", "day"]
    # labels = ["rho", "group ls", "group os", "unit ls", "unit os", "noise"]
    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=4)
    i = 0
    for k, fn in transforms.items():
         mu = priors[k].loc.item()
         s = priors[k].scale.item()
         samples = np.random.normal(mu,s,10000)
         samples = fn(torch.from_numpy(samples)).numpy().reshape(-1)
         if labels[i] in ["noise", "group os", "unit os", "day", "weekday"]:
              samples = np.sqrt(samples)
         sns.histplot(samples, ax=axes[int(i/4), int(i%4)])
         axes[int(i/4)][int(i%4)].legend([labels[i]])
         i = i + 1

    fig.suptitle("Hyperparameter priors")
    plt.savefig("results/pyropriors.png")
     

def plot_prior(model):
    param_list = ["likelihood.noise_covar.noise_prior", "t_covar_module.outputscale_prior", "t_covar_module.base_kernel.lengthscale_prior"]
    xmax = [1,1,60]
    labels = ["noise", "os","ls"]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in range(3):
         parts = param_list[i].split(".")
         prior = model
         for part in parts:
              prior = getattr(prior, part)
         m = prior.concentration.item()
         s = prior.rate.item()
         x = np.linspace(0, xmax[i], 10000)
         # pdf = (np.exp(-(np.log(x) - m)**2 / (2 * s**2)) / (x * s * np.sqrt(2 * np.pi)))
         pdf = x**(m-1)*np.exp(-x*s)*s**m/sps.gamma(m)
         ax[int(i/2)][int(i%2)].plot(x, pdf, color='r', linewidth=2)
         ax[int(i/2)][int(i%2)].legend([labels[i]+" a: " + str(np.around(m,1)) + " b: " + str(np.around(s,1))])

    x = np.linspace(-1,1,10000)
    pdf = 1/2*np.ones(x.shape)
    ax[1][1].plot(x, pdf, color='r', linewidth=2)
    ax[1][1].legend(["rho"])
    fig.suptitle("Gamma prior")
    plt.savefig("results/gammaprior.png")
    plt.close()
    return 

def plot_posterior(chain):
    labels = ["noise", "group ls", "group os", "unit ls", "unit os", "rho"]
    fig, axes = plt.subplots(nrows=2, ncols=3)
    for i in range(6):
        samples = 1/(1+np.exp(-1*getattr(chain, labels[i]).reshape(-1)))
        if i>=2:
            samples = np.sqrt(samples)
        sns.distplot(samples, ax=axes[int(i/2), int(i%2)])
        axes[int(i/2)][int(i%2)].legend([labels[i]])

    fig.suptitle("Gamma posterior")
    plt.savefig("results/gammaposterior.png")
    plt.show()
    return

def plot_pyro_posterior(mcmc_samples, transforms):
#     param_list = ["likelihood.noise_covar.noise_prior","group_t_covar_module.base_kernel.lengthscale_prior",
#      "group_t_covar_module.outputscale_prior", "unit_t_covar_module.base_kernel.lengthscale_prior",
#      "unit_t_covar_module.outputscale_prior", "task_covar_module.rho_prior"]
    labels = ["rho", "group ls", "group os", "unit ls", "unit os", "noise","weekday", "day"]
    # labels = ["rho", "group ls", "group os", "unit ls", "unit os", "noise"]
    fig, axes = plt.subplots(figsize=(20, 10), nrows=2, ncols=4)
    i = 0
    for k, fn in transforms.items():
         samples = fn(mcmc_samples["model$$$" + k]).numpy().reshape(-1)
         if labels[i] in ["noise", "group os", "unit os", "day", "weekday"]:
              samples = np.sqrt(samples)
         sns.histplot(samples, ax=axes[int(i/4), int(i%4)])
         axes[int(i/4)][int(i%4)].legend([labels[i]])
         i = i + 1

    fig.suptitle("Hyperparameter posterior")
    plt.savefig("results/gammaposterior.png")

#     fig, axes = plt.subplots(nrows=3, ncols=3)
#     for i in range(9):
#          s = 100*(i)
#          e = 100*(i+1)
#          samples = mcmc_samples[param_list[3]].numpy().reshape(-1)[s:e]
#          sns.distplot(samples, ax=axes[int(i/3), int(i%3)])
#          axes[int(i/3)][int(i%3)].legend([str(s)+":"+str(e)])

#     fig.suptitle("Rho posterior")
#     plt.savefig("results/gammaposteriorrho.png")

    return 


def visualize(test_t, X_tr, Y_tr, m_tr, lower_tr, upper_tr, X_co, Y_co, m_co, lower_co, upper_co, ATT, T0):
     # Plot unit-level treatment effect
    f, axs = plt.subplots(2, 2, figsize=(12, 6))

    N_tr, T= list(Y_tr.shape)
    N_co = list(X_co.shape)[0]
    d = list(X_tr.shape)[2] - 1

    ax_plot(axs[0], test_t, X_tr, Y_tr, m_tr, lower_tr, upper_tr, "Treated")
    ax_plot(axs[1], test_t, X_co, Y_co, m_co, lower_co, upper_co, "Control")

    for i in range(2):
        for j in range(2):
             axs[i][j].axvline(x=T0, color='red', linewidth=1.0)
    plt.savefig("results/units.png")
    plt.show()


    # Plot group-level treatment effect
    # Initialize plots
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    ax1.plot(1+test_t.detach().numpy(), torch.mean(Y_tr, dim=0).detach().numpy(), 'k', linewidth=1.0, label='Treated Averaged')

    # Averaged predictive mean
    ax1.plot(1+test_t.detach().numpy(), torch.mean(m_tr, dim=0).detach().numpy(), 'k--', linewidth=1.0, label='Estimated Y(0) Average for the Treated')

    for i in range(N_tr):
        # Plot training data 
        ax1.plot(1+X_tr[i,:,-1].detach().numpy(), Y_tr[i,:].detach().numpy(),\
             color='grey', alpha=0.8, label='Treated' if i==0 else None)
    
    for i in range(N_co):
        # Plot training data 
        ax1.plot(1+X_co[i,:,-1].detach().numpy(), Y_co[i,:].detach().numpy(),\
             color='grey', alpha=0.2, label='Control' if i==0 else None)
    
    # Treatment Time
    ax1.axvline(x=T0, color='red', linewidth=1.0)

    ax1.legend(loc=0)

    # Estimated ATT
    ax2.plot(1+test_t.detach().numpy(), torch.mean(Y_tr, dim=0).detach().numpy() - torch.mean(m_tr, dim=0).detach().numpy(),\
         'k', linewidth=1.0, label='Estimated ATT')
        
    # True ATT
    ax2.plot(1+test_t.detach().numpy(), torch.mean(ATT, dim=0).detach().numpy(), 'k--', linewidth=1.0, label='True ATT')

    # Shaded area for critical interval
    ax2.fill_between(1+test_t.detach().numpy(), torch.mean(Y_tr, dim=0).detach().numpy() - torch.mean(upper_tr, dim=0).detach().numpy(),\
         torch.mean(Y_tr, dim=0).detach().numpy() - torch.mean(lower_tr, dim=0).detach().numpy(), alpha=0.5, label="95% Critical Interval")

    ax2.axvline(x=T0, color='red', linewidth=1.0)
    ax2.legend(loc=0)

    plt.savefig("results/synthetic.png")
    plt.show()


def visualize_synthetic(X_tr, X_co, Y_tr, Y_co, ATT, model, likelihood, T0):
    # Set into eval mode
    model.eval()
    likelihood.eval()

    N_tr, T= list(Y_tr.shape)
    N_co = list(X_co.shape)[0]
    d = list(X_tr.shape)[2] - 1

    test_x_tr = X_tr.reshape(-1,d+1)
    test_x_co = X_co.reshape(-1,d+1)
    test_y_tr = Y_tr.reshape(-1)
    test_y_co = Y_co.reshape(-1)
    test_t = X_tr[0,:,-1] # torch.arange(T, dtype=torch.float)

    test_i_tr = torch.full_like(test_y_tr, dtype=torch.long, fill_value=1)
    test_i_co = torch.full_like(test_y_co, dtype=torch.long, fill_value=0)    

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
         f_pred_tr = model(test_x_tr, test_i_tr)
         f_pred_co = model(test_x_co, test_i_co)

    # Get lower and upper confidence bounds
    lower_tr, upper_tr = f_pred_tr.confidence_region()
    m_tr = f_pred_tr.mean.reshape(N_tr, T)
    lower_tr = lower_tr.reshape(N_tr, T)
    upper_tr = upper_tr.reshape(N_tr, T)

    lower_co, upper_co = f_pred_co.confidence_region()
    m_co = f_pred_co.mean.reshape(N_co, T)
    lower_co = lower_co.reshape(N_co, T)
    upper_co = upper_co.reshape(N_co, T)

    visualize(test_t, X_tr, Y_tr, m_tr, lower_tr, upper_tr, X_co, Y_co, m_co, lower_co, upper_co, ATT, T0)

def visualize_localnews(data, test_x, test_y, test_g, model, model2, likelihood, T0, station_le, train_condition):
    # Set into eval mode

    model.eval()
    model2.eval()
    likelihood.eval()
    for i in range(len(model.x_covar_module)):
        model.x_covar_module[i].c2 = torch.tensor(0.0**2)
        model2.x_covar_module[i].c2 = torch.tensor(0.0**2)

    with torch.no_grad(), gpytorch.settings.prior_mode(True), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    K_sum = copy.deepcopy(f_pred.covariance_matrix.detach().numpy())
    mu_sum = copy.deepcopy(f_pred.mean.detach().numpy())

    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "m": f_pred.mean})
    result = result.groupby(['t','g'], as_index=False)[['m']].mean()
    m_1 = result[result.g==1].m.to_numpy()

    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model2(test_x)

    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "m": f_pred.mean})
    result = result.groupby(['t','g'], as_index=False)[['m']].mean()
    m_0 = result[result.g==1].m.to_numpy()

    with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
         f_pred = model2(test_x)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()

    station_ids = data.station_id.unique()
    
    for station_id in []:
         mask = (data.station_id==station_id).to_numpy()
         test_t = test_x[mask, -1]
         idx = np.argsort(test_t)
         test_t = test_t[[idx]]
         lower_i = lower[mask][idx]
         upper_i = upper[mask][idx]
         m_i = f_pred.mean[mask][idx]
         treatment = data[mask].sinclair2017.unique()[0]
         LABEL = "treated" if treatment else "control"
         y_i = test_y[mask][[idx]]

         plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(1+test_t.detach().numpy(), y_i.detach().numpy(),\
               color='blue', s=4, label=LABEL + " " + str(station_id))
         plt.plot(1+test_t.detach().numpy(), m_i.detach().numpy(),\
               'k--', linewidth=2.0, label='Estimated Y(0)')
         plt.fill_between(1+test_t.detach().numpy(), lower_i.detach().numpy(),\
               upper_i.detach().numpy(), alpha=0.3)
         plt.legend(loc=2)
         plt.title("{} Unit {}".format(LABEL, station_id))
         plt.axvline(x=T0, color='red', linewidth=1.0)
         plt.savefig("results/localnews_{}.png".format(station_id))
         plt.close()

    model.unit_t_covar_module.outputscale = 0
    model.unit_mean_module.constantvector.data.fill_(0.0)
    model2.unit_t_covar_module.outputscale = 0
    model2.unit_mean_module.constantvector.data.fill_(0.0)

     # Make predictions
    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
         f_pred = model2(test_x)

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
    fill_alpha = [0.1, 0.2]
    mean_color = ["blue", "tomato"]
    y_color = ["purple", "crimson"]
    # plt.rcParams["figure.figsize"] = (15,5)
    for g in []:
         test_t = np.unique(result[result.g==g].t)
         lower_g = result[result.g==g].lower.to_numpy()
         upper_g = result[result.g==g].upper.to_numpy()
         m_g = result[result.g==g].m.to_numpy()
         if g==0:
              m_g_0 = m_g
              v_0 = (upper_g - lower_g)/2
         else:
              m_g_1 = m_g
              v_1 = (upper_g - lower_g)/2
         y_g = result[result.g==g].y.to_numpy()
         LABEL = "Acquired" if g==1 else "Not Acquired"
         plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(x=1+test_t, y=y_g, c=y_color[0], s=4, label=LABEL + " avg")
         plt.plot(1+test_t, m_g, c=mean_color[0], linewidth=2, label=LABEL +' estimated Y(0)')
         plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[g], label=LABEL + " 95% CI")
         plt.legend(loc=2)
         plt.title("Averaged " + LABEL + " Group Trends ")
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
         plt.savefig("results/localnews_MAP_{}.png".format(LABEL))
         plt.close()

#     plt.savefig("results/localnews_MAP.png")
#     plt.close()


 # Make predictions
    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
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
    fill_alpha = [0.1, 0.2]
    mean_color = ["blue", "tomato"]
    y_color = ["purple", "crimson"]
    plt.rcParams["figure.figsize"] = (15,5)
    for g in []:
         test_t = np.unique(result[result.g==g].t)
         lower_g = result[result.g==g].lower.to_numpy()
         upper_g = result[result.g==g].upper.to_numpy()
         m_g = result[result.g==g].m.to_numpy()
     #     if g==0:
     #          m_g_0 = m_g
     #          v_0 = (upper_g - lower_g)/2
     #     else:
     #          m_g_1 = m_g
     #          v_1 = (upper_g - lower_g)/2
         y_g = result[result.g==g].y.to_numpy()
         LABEL = "Acquired" if g==1 else "Not Acquired"
         # plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(x=1+test_t, y=y_g, c=y_color[g], s=4, label=LABEL + " avg")
         plt.plot(1+test_t, m_g, c=mean_color[g], linewidth=2, label=LABEL +' estimated Y')
         plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[g], label=LABEL + " 95% CI")
         plt.legend(loc=2)
         plt.title("Averaged " + LABEL + " Group Trends ")
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")

#     plt.savefig("results/localnews_MAP.png")
#     plt.close()

    model.group_t_covar_module.outputscale = 0
    model.group_mean_module.constantvector.data.fill_(0.0)

    with torch.no_grad(), gpytorch.settings.prior_mode(True), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    K_D = copy.deepcopy(f_pred.covariance_matrix.detach().numpy())
    mu_D = copy.deepcopy(f_pred.mean.detach().numpy())

    # model.group_t_covar_module.outputscale = 0.0
    # model.unit_mean_module.constantvector.data.fill_(0.0) 
    # model.group_mean_module.constantvector.data.fill_(0.0) 

     # Make predictions
#     with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
#          f_pred = model(test_x)

#     # Get lower and upper confidence bounds
#     lower, upper = f_pred.confidence_region()

#     result = pd.DataFrame({
#          "t":test_x[:,-1],
#          "g": test_g,
#          "upper": upper,
#          "lower": lower,
#          "m": f_pred.mean,
#          "y": test_y})
#     result = result.groupby(['t','g'], as_index=False)[['lower','upper','m','y']].mean()

#     for g in [1]:
#          test_t = np.unique(result[result.g==g].t)
#          lower_g = result[result.g==g].lower.to_numpy()
#          upper_g = result[result.g==g].upper.to_numpy()
#          m_g = result[result.g==g].m.to_numpy()
#          v = (upper_g - lower_g)/2

#          plt.rcParams["figure.figsize"] = (15,5)
#          plt.plot(1+test_t, m_g, c=mean_color[0], linewidth=2, label='treatment effect')
#          plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[1], label="95% CI")
#          plt.legend(loc=2)
#          plt.title("Treatment Effect Trend ")
#          plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
#          plt.savefig("results/localnews_MAP_effect.png")
#          plt.close()


#     plt.plot(test_t,m_g_1-m_g_0[2:])
#     plt.plot(test_t,m_g)
#     plt.show()

     # verify conditioning on sum of two GPs
    g = 1
    mu_p = mu_D+ K_D.dot(np.linalg.inv(K_sum+np.identity(K_sum.shape[0])*likelihood.noise.item())).dot(test_y-mu_sum)
    K_p = K_D - K_D.dot(np.linalg.inv(K_sum+np.identity(K_sum.shape[0])*likelihood.noise.item())).dot(K_D)

    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "m": mu_p,
         "s2": K_p.diagonal(),
         "y": test_y})
    result = result.groupby(['t','g'], as_index=False)[['m','y',"s2"]].mean()

    m_g = result[result.g==g].m.to_numpy()
    test_t = np.unique(result[result.g==g].t)
    plt.rcParams["figure.figsize"] = (15,5)
    plt.plot(1+test_t, m_g, c="darkblue", linewidth=2, label='treatment effect')

    std_p = np.sqrt(result[result.g==g].s2.to_numpy())
    lower_g = m_g - 1.96*std_p
    upper_g = m_g + 1.96*std_p
    plt.plot(1+test_t,m_1-m_0,  c="blue", linestyle="--",linewidth=1, label='estimated Y(1)-Y(0)')
    plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[1], label="95% CI")
    plt.legend(loc=2)
    plt.title("Treatment Effect Trend ")
    plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
    plt.savefig("results/localnews_MAP_effect.png")
    plt.close()
    # plt.show()


def visualize_localnews_MCMC(data, train_x, train_y, train_i, test_x, test_y, test_i, model,\
                likelihood, T0, station_le, num_samples):
    # Set into eval mode
    model.eval()
    likelihood.eval()

    for i in range(1,len(model.x_covar_module)):
        model.x_covar_module[i].c2 = torch.tensor(0.0**2)

    T = list(torch.unique(test_x[:,-1]).shape)[0]
    N = torch.unique(train_i).shape[0]
    d = list(train_x.shape)[1] - 1
    # test_t = torch.arange(T, dtype=torch.float)

    expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
    expanded_test_i = test_i.unsqueeze(1).repeat(num_samples, 1, 1)
    expanded_test_y = test_y.unsqueeze(1).repeat(num_samples, 1, 1)
    expanded_test_D = torch.tensor(data.sinclair2017.to_numpy()).double().unsqueeze(1).repeat(num_samples, 1, 1)
    sample_ids = torch.cat([torch.ones(test_y.shape[0])*i for i in range(num_samples)])

     # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
         f_pred = model(expanded_test_x, expanded_test_i)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()

    result = pd.DataFrame({
         "sample_id": sample_ids,
         "t": expanded_test_x[:,:,-1].reshape(-1),
         "i": expanded_test_i.reshape(-1),
         "upper": upper.reshape(-1),
         "lower": lower.reshape(-1),
         "m": f_pred.mean.reshape(-1),
         "y": expanded_test_y.reshape(-1),
         "sinclair2017": expanded_test_D.reshape(-1)})
    result = result.groupby(['t','i','sample_id'], as_index=False)['lower','upper','m','y'].mean()
    fill_alpha = [0.2, 0.5]
    mean_color = ["blue", "slateblue"]
    y_color = ["purple", "deeppink"]
    for i in [0,1]:
         test_t = np.unique(result[result.i==i].t)
         lower_i = result[result.i==i].groupby(["t"], as_index=False)['lower'].mean().lower.to_numpy()
         upper_i = result[result.i==i].groupby(["t"], as_index=False)['upper'].mean().upper.to_numpy()
         m_i = result[result.i==i].groupby(["t"], as_index=False)['m'].mean().m.to_numpy()
         y_i = result[result.i==i].groupby(["t"], as_index=False)['y'].mean().y.to_numpy()
         LABEL = "Acquired" if i==1 else "Not Acquired"

         plt.scatter(x=1+test_t, y=y_i, c=y_color[i], s=1, label=LABEL + " avg")
         plt.plot(1+test_t, m_i, c=mean_color[i], linewidth=0.5, label=LABEL +' estimated Y(0)')
         plt.fill_between(1+test_t, lower_i, upper_i, color='grey', alpha=fill_alpha[i], label=LABEL + " 95% CI")
         # plt.legend(loc=2)
         plt.title("Averaged " + LABEL + " Group Trends ")
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
         plt.savefig("results/localnews_MCMC_{}.png".format(LABEL))
         plt.close()

