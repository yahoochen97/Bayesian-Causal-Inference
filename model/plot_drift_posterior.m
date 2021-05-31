function [fig, results] = plot_drift_posterior(theta,mean_function,...
    covariance_function, non_drift_idx, x, y)
%
%   Condition the drift process on the sum of counterfactual+effect
%   observatoins and plot the posterior of drift process.
%   
%   Inputs:
%       - theta: hyperparameters of the full model
%       - mean_function: mean function of the full model
%       - covariance_function: covariance function of the full model
%       - non_drift_idx: index of non-drift output scale/variance
%       parameters (e.g. [2, 5, 7, 10, 12])
%       - x: n by D matrix of training inputs
%       - y: column vector of length n of training targets
% 
%   Outputs:
%       - fig: drift posterior plot
%       - results: drift posterior table
%

    % drift process prior
    theta_drift = theta.cov;
    theta_drift(non_drift_idx) = log(0);
    m_drift = feval(mean_function{:}, theta.mean, x)*0;
    K_drift = feval(covariance_function{:}, theta_drift, x);

    % sum of counterfactual+effect observations
    theta_sum = theta.cov;
    % theta_sum([5, 10,12]) = log(0);
    m_sum = feval(mean_function{:}, theta.mean, x);
    K_sum = feval(covariance_function{:}, theta_sum, x);

    % drift posterior
    V = K_sum+exp(2*theta.lik)*eye(size(K_sum,1));
    inv_V = pinv(V);
    m_post = m_drift + K_drift*inv_V*(y-m_sum);
    K_post = K_drift - K_drift*inv_V*K_drift;

    % average over group
    results = table;
    results.m = m_post(x(:,end)~=0,:);
    results.day = x(x(:,end)~=0,1);
    tmp = diag(K_post);
    results.s2 = tmp(x(:,end)~=0,:);
    results.y = y(x(:,end)~=0,:);
    results = groupsummary(results, 'day', 'mean');
    mu = results.mean_m;
    s2 = results.mean_s2;
    days = results.day;
    counts = results.GroupCount;

    fig = figure(1);
    clf;
    f = [mu+2*sqrt(s2/counts); flip(mu-2*sqrt(s2/counts),1)];
    fill([days; flip(days,1)], f, [7 7 7]/8);
    hold on; plot(days, mu);

end

