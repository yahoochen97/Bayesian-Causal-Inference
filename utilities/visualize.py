import torch
import gpytorch
from matplotlib import pyplot as plt


def visualize(X_tr, X_co, Y_tr, Y_co, ATT, model, likelihood):
    # Set into eval mode
    model.eval()
    likelihood.eval()

    # Initialize plots
    # task tr: treatment
    # task co: control

    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    N_tr, T= list(Y_tr.shape)
    N_co = list(X_co.shape)[0]
    d = list(X_tr.shape)[2] - 1
    T0 = 20

    test_x_tr = X_tr.reshape(-1,d+1)
    test_x_co = X_co.reshape(-1,d+1)
    test_y_tr = Y_tr.reshape(-1)
    test_y_co = Y_co.reshape(-1)
    test_t = X_tr[0,:,-1] # torch.arange(T, dtype=torch.float)

    test_i_tr = torch.full_like(test_y_tr, dtype=torch.long, fill_value=0)
    test_i_co = torch.full_like(test_y_co, dtype=torch.long, fill_value=1)    

    # Make predictions
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
         f_pred_tr = model(test_x_tr, test_i_tr)

    # Get lower and upper confidence bounds
    lower_tr = f_pred_tr.mean - 1.92*torch.sqrt(f_pred_tr.variance)
    upper_tr = f_pred_tr.mean + 1.92*torch.sqrt(f_pred_tr.variance)
    m_tr = f_pred_tr.mean.reshape(N_tr, T)
    lower_tr = lower_tr.reshape(N_tr, T)
    upper_tr = upper_tr.reshape(N_tr, T)

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
