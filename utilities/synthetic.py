import math
import torch

def generate_synthetic_data(N_tr, N_co, T, T0, d, Delta, noise_std, seed):
    '''
        Generate synthetic data.
    Inputs:
        - N_tr: number of treatment units
        - N_co: number of control units
        - T: total time
        - T0: treatment time
        - d: dimension of covariates
        - Delta: slope of homogeneous treatment effect
        - noise_std: general noise std
        - seed: seed for replication
    Outputs:
        - X_tr: time-dependent covariates for treatment group, N_tr*T*(d+1) tensor, last column is time
        - X_co: time-dependent covariates for control group, N_co*T*(d+1) tensor, last column is time
        - Y_tr: observations for treatment group, N_tr*T tensor
        - Y_co: observations for control group, N_co*T tensor
        - T_tr: N_tr*1 tensor, treatment time for treatment units
    '''
    torch.manual_seed(seed)

    # generate time series
    # here assume x_itd = 1+a*b+a+b+e
    # confounder: a ~ N(0,1)
    # loading: b_co ~ U[-1,1], b_tr ~ U[-0.6, 1.4]
    # error: e ~ N(0,noise_std)
    train_t = torch.arange(T, dtype=torch.float)

    X_tr = torch.randn(N_tr, T, d) * noise_std
    X_co = torch.randn(N_co, T, d) * noise_std

    a = torch.randn(T,d)
    b_co = 2*torch.rand(N_co,d) - 1
    b_tr = 2*torch.rand(N_tr,d) - 0.6

    for i in range(N_tr):
        for t in range(T):
            for k in range(d):
                X_tr[i, t, k] += 1 + a[t,k] + b_tr[i,k] + a[t,k]*b_tr[i,k]

    for i in range(N_co):
        for t in range(T):
            for k in range(d):
                X_co[i, t, k] += 1 + a[t,k] + b_co[i,k] + a[t,k]*b_co[i,k]

    # here assume y_it = delta*D + sum((d+1)*x_itd) + alpha_t + beta + e
    # alpha_tr = [sin(t) + 2*t]/5 + e
    # alpha_co = [cos(t) + t]/5 + e
    # beta_co ~ U[-1,1], beta_tr ~ U[-0.6, 1.4]
    # e ~ N(0, noise_std)
    Y_tr = torch.randn(N_tr, T) * noise_std
    Y_co = torch.randn(N_co, T) * noise_std

    alpha_tr = (torch.sin(train_t) + 2*train_t)/5 + torch.randn(train_t.size()) * noise_std
    alpha_co = (torch.cos(train_t) + train_t)/5 + torch.randn(train_t.size()) * noise_std
    beta_co = 2*torch.rand(N_co,d) - 1
    beta_tr = 2*torch.rand(N_co,d) - 0.6

    for i in range(N_tr):
        for t in range(T):
            Y_tr[i,t] += alpha_tr[t]
            for k in range(d):
                Y_tr[i,t] += (k+1)*X_tr[i,t,k] + beta_tr[i,k]

    for i in range(N_co):
        for t in range(T):
            Y_co[i,t] += alpha_co[t]
            for k in range(d):         
                Y_co[i,t] += (k+1)*X_co[i,t,k] + beta_co[i,k]

    # ATT matrix
    ATT = torch.zeros(Y_tr.shape)
        
    ATT[:,T0:] += Delta*(train_t[T0:]-T0) # + torch.randn(ATT.size())[:,T0:] * noise_std
    Y_tr = Y_tr + ATT

    X_tr = torch.cat([X_tr, torch.unsqueeze(train_t.expand(N_tr, T),dim=2)], dim=2)
    X_co = torch.cat([X_co, torch.unsqueeze(train_t.expand(N_co, T),dim=2)], dim=2)

    return X_tr, X_co, Y_tr, Y_co, ATT

