from __future__ import absolute_import, division, print_function
import sys
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append('../../code/causal_multitask_gaussian_processes_ite')

from utils.metrics import compute_PEHE, mean_confidence_interval
from models.causal_models import CMGP

def load_synthetic_data(data, true_effects):
    
    # Read the covariates and treatment assignments from the original study
    # ----------------------------------------------------------------------
    X              = np.array(data[['weekday', 'day']])
    Group          = np.array(data['group'])
    W              = np.array(data['D'])
    Y              = np.array(data['y'])
    Day            = np.array(data['day'])
    
    TE             = true_effects[Day-1]
    Y_0            = Y - TE*W
    Y_1            = Y + TE*(1-W)
    Y              = np.transpose(np.array([W, Y, TE]))

    # Prepare the output dataset 
    # --------------------------
    DatasetX       = pd.DataFrame(X,columns='Weekday Day'.split())
    DatasetY       = pd.DataFrame(Y,columns='Treatment Response TE'.split())
    Dataset        = DatasetX.join(DatasetY)
    
    Dataset['Y_0'] = Y_0
    Dataset['Y_1'] = Y_1
    
    return Dataset


def sample_data(data, true_effects):
    Dataset     = load_synthetic_data(data, true_effects)
    feat_name   = 'Weekday Day'
    
    X_train       = np.array(Dataset[feat_name.split()])
    W_train       = np.array(Dataset['Treatment'])
    Y_train       = np.array(Dataset['Response'])
    T_true_train  = np.array(Dataset['TE'])
    Y_cf_train    = np.array(Dataset['Treatment'] * Dataset['Y_0'] + (1- Dataset['Treatment']) * Dataset['Y_1'])

    Y_0_train     = np.array(Dataset['Y_0'])
    Y_1_train     = np.array(Dataset['Y_1'])
    train_data    = (X_train, W_train, Y_train, Y_0_train, Y_1_train, Y_cf_train, T_true_train)
    
    return train_data 


def run_experiment(data, true_effects, mode="CMGP"):
    
    train_data                              = sample_data(data, true_effects)
    X_train, W_train, Y_train, T_true_train = train_data[0], train_data[1], train_data[2], train_data[6]
    
    model = CMGP(dim=2, mode=mode)

    model.fit(X_train, Y_train, W_train)
    TE_est_train, _, _, var_all = model.predict(X_train)
    PEHE_train = compute_PEHE(TE_est_train, T_true_train)
    Day = train_data[0][:,1]
    T = int(max(Day))
    N = var_all.shape[0]/T

    TE_est = np.zeros((T,))
    TE_var = np.zeros((T,))
    for i in range(T):
        TE_est[i] = np.mean(TE_est_train[Day==(i+1)])
        TE_var[i] = np.mean(var_all[Day==(i+1)])/N

    return TE_est, np.sqrt(TE_var), PEHE_train

def main():
    start = time.time()
    data = pd.read_csv("localnewsdata.csv")
    T = data.day.max()
    effect_file = "./localnewseffects.csv"
    effects = pd.read_csv(effect_file)
    true_effects = np.zeros((T,))
    days = effects['day'].values
    effects = effects['mean'].values
    for t in range(1,T+1):
        if t in days:
            true_effects[t-1]=effects[days==t]

    for t in range(90,T+1):
        if true_effects[t-1]==0:
            t_ = t + 1 
            while true_effects[t_-1]==0:
                t_ += 1    
            true_effects[t-1]=true_effects[t_-1]

    # result_file = "./localnewsalleffects.csv"
    # result = pd.DataFrame({"effect": true_effects, "day": days})
    # result.to_csv(result_file ,index=False)

    TE_est, std_est, PEHE_train = run_experiment(data, true_effects)
    
    result_file = "../results/localnews_cmgp_.csv"
    result = pd.DataFrame({"mu" :TE_est, "std": std_est})
    result.to_csv(result_file ,index=False)
    end = time.time()
    print(end - start)

if __name__ == "__main__":
    main()
    