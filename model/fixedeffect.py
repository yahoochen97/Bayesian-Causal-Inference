from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import summary_table
import pandas as pd
import numpy as np
import torch
from utilities.visualize import ax_plot
from matplotlib import pyplot as plt


def TwoWayFixedEffectModel(X_tr, X_co, Y_tr, Y_co, ATT, T0):
    N_tr, T, d = list(X_tr.shape)
    N_co = list(X_co.shape)[0]
    x = torch.cat([X_tr.reshape(-1,d),X_co.reshape(-1,d)]).detach().numpy()

    units = np.concatenate([np.tile(np.array([i for i in range(N_tr)]), T),
                            np.tile(np.array([i+N_tr for i in range(N_co)]), T)]).reshape(-1,1)

    treated = np.logical_and((units<N_tr), (x[:,-1]>=T0).reshape(-1,1)).astype("float").reshape(-1,1)
    y = torch.cat([Y_tr.reshape(-1,1),Y_co.reshape(-1,1)]).detach().numpy()
    COLUMNS = ['x{}'.format(i+1) for i in range(d-1)]
    COLUMNS.append("time")
    COLUMNS.append("y")
    COLUMNS.append("unit")
    COLUMNS.append("treated")
    data = pd.DataFrame(np.concatenate((x,y,units,treated),axis=1),columns=COLUMNS)
    
    # x1 + x2 + x3 + x4 + x5 +
    fit = ols('y ~ ' + " ".join(['x{} +'.format(i+1) for i in range(d-1)]) + 
            ' C(time) + C(unit) + treated:C(time)', data=data).fit() 

    ypred = fit.predict(data)
    m_tr = torch.tensor(ypred[:N_tr*T].to_numpy().reshape(N_tr,T))
    m_co = torch.tensor(ypred[N_tr*T:].to_numpy().reshape(N_co,T))

    for t in range(T0, T, 1):
        m_tr[:, t] -= fit.params["treated:C(time)[{}.0]".format(t)]

    _, data, _ = summary_table(fit, alpha=0.05)

    predict_mean_ci_lower, predict_mean_ci_upper = data[:, 4:6].T

    lower_tr = torch.tensor(predict_mean_ci_lower[:N_tr*T].reshape(N_tr,T))
    upper_tr = torch.tensor(predict_mean_ci_upper[:N_tr*T].reshape(N_tr,T))
    lower_co = torch.tensor(predict_mean_ci_lower[N_tr*T:].reshape(N_co,T))
    upper_co = torch.tensor(predict_mean_ci_upper[N_tr*T:].reshape(N_co,T))

    for t in range(T0, T, 1):
        lower_tr[:, t] -= fit.conf_int().loc["treated:C(time)[{}.0]".format(t),1]
        upper_tr[:, t] -= fit.conf_int().loc["treated:C(time)[{}.0]".format(t),0]
        # lower_co[:, t] -= fit.params["treated:C(time)[{}.0]".format(t)]
        # upper_co[:, t] -= fit.params["treated:C(time)[{}.0]".format(t)]

    f, axs = plt.subplots(2, 2, figsize=(12, 6))

    test_t = torch.arange(T, dtype=torch.float)
    
    ax_plot(axs[0], test_t, X_tr, Y_tr, m_tr, lower_tr, upper_tr, "Treated")
    ax_plot(axs[1], test_t, X_co, Y_co, m_co, lower_co, upper_co, "Control")

    for i in range(2):
        for j in range(2):
             axs[i][j].axvline(x=T0, color='red', linewidth=1.0)

    plt.savefig("results/fixedeffectunits.png")
    plt.show()


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

    plt.savefig("results/fixedeffectsynthetic.png")
    plt.show()

    return fit
