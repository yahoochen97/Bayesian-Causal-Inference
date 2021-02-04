import torch
import json
import gpytorch
from model.multitaskmodel import MultitaskGPModel
from utilities.savejson import savejson
from utilities.visualize import visualize_multitask
from utilities.synthetic import generate_synthetic_data
from model.fixedeffect import TwoWayFixedEffectModel
import os

smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 500


def train(train_x, train_i, train_y, model, likelihood, optimizer):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x, train_i)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    return model, likelihood


def main():
    # load configurations
    with open('model/conf.json') as f:
        data = json.load(f)

    N_tr = data["N_tr"]
    N_co = data["N_co"]
    T = data["T"]
    T0 = data["T0"]
    d = data["d"]
    noise_std = data["noise_std"]
    Delta = data["treatment_effect"]
    seed = data["seed"]

    X_tr, X_co, Y_tr, Y_co, ATT = generate_synthetic_data(N_tr, N_co, T, T0, d, Delta, noise_std, seed)

    # fit = TwoWayFixedEffectModel(X_tr, X_co, Y_tr, Y_co, ATT, T0)
    # return 
    
    train_x_tr = X_tr[:,:T0].reshape(-1,d+1)
    train_x_co = X_co.reshape(-1,d+1)
    train_y_tr = Y_tr[:,:T0].reshape(-1)
    train_y_co = Y_co.reshape(-1)

    train_x = torch.cat([train_x_tr, train_x_co])
    train_y = torch.cat([train_y_tr, train_y_co])
    train_i = torch.cat([torch.full_like(train_y_tr, dtype=torch.long, fill_value=0),
                        torch.full_like(train_y_co, dtype=torch.long, fill_value=1)])
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MultitaskGPModel((train_x, train_i), train_y, likelihood)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    if not os.path.isdir("results"):
        os.mkdir("results")

    if False:
        state_dict = torch.load('results/model_state.pth')
        model.load_state_dict(state_dict)
    else:
        model, likelihood = train(train_x, train_i, train_y, model, likelihood, optimizer)
        torch.save(model.state_dict(), 'results/model_state.pth')

    json_file = "results/optimizedhyps.json"
    savejson(model, likelihood, json_file)

    visualize_multitask(X_tr, X_co, Y_tr, Y_co, ATT, model, likelihood, T0)


if __name__ == "__main__":
    main()
