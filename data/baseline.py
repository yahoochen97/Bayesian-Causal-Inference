import numpy as np
import argparse
from matplotlib import pyplot as plt
import os

N_tr = 10
N_co = 10
T = 50
T0 = 40
synthetic_path = 'data/synthetic'
TITLES = ["Non linear but parallel",
            "Linear but not parallel",
            "Non linear and not parallel"]

def f1(x):
    # non linear but parallel
    y_tr = np.cos(x/6) / (T/2)  + x / T / 5 + 0.2
    y_co = y_tr - 0.1
    return y_tr, y_co

def f2(x):
    # linear but not parallel
    y_co = x / T / 6 + 0.2
    y_tr = x / T / 5 + 0.3
    return y_tr, y_co

def f3(x):
    # non linear and not parallel
    y_tr = np.cos(x/6) / (T/2)  + x / T / 5 + 0.2
    y_co = np.cos(x/5) / (T/2)  + x / T / 6 + 0.1
    return y_tr, y_co

fs = [f1, f2, f3]

def generate_data():
    for k in range(len(TITLES)):
        x = np.arange(T)

        treat = np.zeros((N_tr, T))
        control = np.zeros((N_co, T))
        y_tr, y_co = fs[k](x)

        for i in range(N_tr):
            treat[i] = y_tr
            b = np.random.uniform(-0.02,0.02, 1)
            treat[i] += np.random.normal(b, 0.05, T)
            effect = 0.05
            treat[i,T0:] += np.random.normal(effect,0.02, T)[T0:]

        for i in range(N_co):
            control[i] = y_co
            b = np.random.uniform(-0.02,0.02, 1)
            control[i] += np.random.normal(b, 0.05, T)

        np.savetxt(synthetic_path+"/treat{}.csv".format(k), treat, delimiter=",")
        np.savetxt(synthetic_path+"/control{}.csv".format(k), control, delimiter=",")

        plot_synthetic_data(treat, control, fs, effect, k)

def plot_synthetic_data(treat, control, f, effect, k):
    x_plot = np.linspace(0,T,1000)
    x = np.arange(T)
    y_tr, y_co = fs[k](x_plot)
    y_tr[x_plot>=T0] += effect
    # plot true mean
    plt.plot(x_plot, y_tr, 'b--', linewidth=1.0, label='Treat true mean')
    plt.plot(x_plot, y_co, 'r--', linewidth=1.0, label='Control true mean')

    # plot averaged data
    plt.scatter(x, np.mean(treat, axis=0), c="purple", s=4, label='Treat sample averaged')
    plt.scatter(x, np.mean(control, axis=0), c="crimson", s=4, label='Control sample averaged')
    plt.legend(loc=2)
    plt.title(TITLES[k])
    plt.savefig(synthetic_path+"/data{}.png".format(k))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python baseline.py --type data')
    parser.add_argument('-t','--type', help='data', required=True)
    args = vars(parser.parse_args())
    if args['type'] == 'data':
        if not os.path.exists(synthetic_path):
            os.makedirs(synthetic_path)
        generate_data()
    else:
        exit()