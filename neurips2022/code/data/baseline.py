import numpy as np
import argparse
from matplotlib import pyplot as plt
import os
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import summary_table

N_tr = 10
N_co = 10
T = 50
T0 = 40
effect = 0.1
synthetic_path = 'data/synthetic'
TITLES = ["Non linear but parallel",
            "Linear but not parallel",
            "Non linear and not parallel"]

def generate_effect(t, SEED):
    # np.random.seed(SEED)
    true_effects = np.zeros(t.shape)
    true_effects += effect/((T-T0)/2)*(t-T0)*((t>=T0) & (t<=(T0+T)/2))
    true_effects += effect*(t>(T0+T)/2)
    return true_effects + np.random.normal(0, 0.01, true_effects.shape)*((t-T0)>=0)

def f1(x):
    # non linear but parallel
    y_tr = np.cos(x/4) / (T/2)  + x / T / 4 + 0.5
    y_co = y_tr - 0.1
    return y_tr, y_co

def f2(x):
    # linear but not parallel
    y_co = x / T / 3 + 0.2
    y_tr = x / T / 5 + 0.4
    return y_tr, y_co

def f3(x, SEED):
    # np.random.seed(SEED)
    coef_a = np.random.normal(1/T**2, 1/T**2/5, size=2)
    coef_b = np.random.normal(2*T/3, T/5, size=2)
    # non linear and not parallel
    y_tr = np.cos(x/5) / (T/2) + coef_a[0]*(x-coef_b[0])**2 + 0.3
    y_co = np.cos(x/4) / (T/2) + coef_a[1]*(x-coef_b[1])**2 + 0.1
    return y_tr, y_co

fs = [f1, f2, f3]
fs = [f3]

TITLES=['quadratic']

def generate_data(SEED):
    np.random.seed(SEED)
    for k in range(len(TITLES)):
        x = np.arange(T)
        treat = np.zeros((N_tr, T))
        control = np.zeros((N_co, T))
        y_tr, y_co = fs[k](x, SEED)
        print(np.corrcoef(y_tr,y_co))

        ATT = np.zeros(treat.shape)
        for i in range(N_tr):
            treat[i] = y_tr
            b = np.random.uniform(-0.05,0.05, 1)
            treat[i] += np.random.normal(b, 0.05, T)
            ATT[i] += generate_effect(x, SEED)
            treat[i] += ATT[i]

        for i in range(N_co):
            control[i] = y_co
            b = np.random.uniform(-0.05,0.05, 1)
            control[i] += np.random.normal(b, 0.05, T)

        np.savetxt(synthetic_path+"/treat_{}.csv".format(SEED), treat, delimiter=",")
        np.savetxt(synthetic_path+"/control_{}.csv".format(SEED), control, delimiter=",")
        np.savetxt(synthetic_path+"/effect_{}.csv".format(SEED), ATT, delimiter=",")

        plot_synthetic_data(treat, control, k, SEED, y_tr, y_co, ATT)
        fixed_effect(treat, control, k, SEED)


def plot_synthetic_data(treat, control, k, SEED,y_tr, y_co, ATT):
    plt.rcParams["figure.figsize"] = (10,5)
    # x_plot = np.linspace(0,T,2*T)
    x_plot = x = np.arange(T)
    # y_tr, y_co = fs[k](x_plot, SEED)
    # y_tr += generate_effect(x_plot, SEED)
    y_tr += np.mean(ATT, axis=0)
    # plot true mean
    plt.plot(x_plot, y_tr, 'b--', linewidth=1.0, label='Treat true mean')
    plt.plot(x_plot, y_co, 'r--', linewidth=1.0, label='Control true mean')

    # plot averaged data
    plt.scatter(x, np.mean(treat, axis=0), c="purple", s=4, label='Treat sample averaged')
    plt.scatter(x, np.mean(control, axis=0), c="crimson", s=4, label='Control sample averaged')
    plt.legend(loc=2)
    plt.title(TITLES[k])
    plt.savefig(synthetic_path+"/data_{}.png".format(SEED))
    plt.close()


def fixed_effect(treat, control, k, SEED):
    x = np.arange(T)
    x = np.concatenate([x for _ in range(N_tr+N_co)]).reshape(-1,1)
    units = np.concatenate([[i for _ in range(T)] for i in range(N_tr+N_co)]).reshape(-1,1)
    treated = np.logical_and((units<N_tr), (x>=T0)).astype("float")
    y = np.concatenate([treat.reshape(-1,1),control.reshape(-1,1)])
    COLUMNS = ["time", "y", "unit", "treated"]
    data = pd.DataFrame(np.concatenate((x,y,units,treated),axis=1),columns=COLUMNS)
    data.to_csv(synthetic_path+"/data_{}.csv".format(SEED), index=False)

    return
    
    fit = ols('y ~ 1 + C(time) + C(unit) + treated:C(time)', data=data).fit()
    ypred = fit.predict(data)
    m_tr = ypred[:N_tr*T].to_numpy().reshape(N_tr,T)
    m_co = ypred[N_tr*T:].to_numpy().reshape(N_co,T)

    # print(fit.summary())

    for t in range(T0, T, 1):
        m_tr[:, t] -= fit.params["treated:C(time)[{}.0]".format(t)]

    _, data, _ = summary_table(fit, alpha=0.05)

    predict_mean_ci_lower, predict_mean_ci_upper = data[:, 4:6].T

    lower_tr = predict_mean_ci_lower[:N_tr*T].reshape(N_tr,T)
    upper_tr = predict_mean_ci_upper[:N_tr*T].reshape(N_tr,T)
    lower_co = predict_mean_ci_lower[N_tr*T:].reshape(N_co,T)
    upper_co = predict_mean_ci_upper[N_tr*T:].reshape(N_co,T)

    for t in range(T0, T, 1):
        lower_tr[:, t] -= fit.conf_int().loc["treated:C(time)[{}.0]".format(t),1]
        upper_tr[:, t] -= fit.conf_int().loc["treated:C(time)[{}.0]".format(t),0]

    test_t = np.arange(T)

    # plt.plot(test_t, np.mean(control, axis=0), color='grey', alpha=0.8)
    # plt.plot(test_t,  np.mean(m_co, axis=0), 'k--', linewidth=1.0, label='Estimated Y(0)')
    # plt.fill_between(test_t, np.mean(lower_co, axis=0), np.mean(upper_co, axis=0), alpha=0.5)
    # plt.show()

    ATT = np.stack([np.mean(treat-m_tr, axis=0),
                    np.mean(treat-upper_tr, axis=0),
                    np.mean(treat-lower_tr, axis=0)])

    plt.rcParams["figure.figsize"] = (15,5)
    plt.plot(test_t, ATT[0],'k--', linewidth=1.0, label="Estimated ATT")
    plt.fill_between(test_t, ATT[1], ATT[2], alpha=0.5, label="ATT 95% CI")
    plt.legend(loc=2)
    plt.savefig(synthetic_path+"/fixedeffect{}_{}.png".format(k, SEED))
    plt.close()
    
    np.savetxt(synthetic_path+"/fixedeffect{}_{}.csv".format(k, SEED), ATT, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python baseline.py --type data')
    parser.add_argument('-t','--type', help='data/twoway', required=True)
    parser.add_argument('-s','--seed', help='seed', required=True)
    args = vars(parser.parse_args())
    SEED = int(args['seed'])
    if args['type'] == 'data':
        if not os.path.exists(synthetic_path):
            os.makedirs(synthetic_path)
        generate_data(SEED)
    else:
        exit()