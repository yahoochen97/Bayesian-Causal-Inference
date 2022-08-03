function [mu, s2, days, counts]=drift_posterior(theta, non_drift_idx,...
    mean_function, covariance_function, x, y, day_index)
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
    inv_V = inv(V);
    m_post = m_drift + K_drift*inv_V*(y-m_sum);
    K_post = K_drift - K_drift*inv_V*K_drift;

    % average over group
    results = table;
    results.m = m_post(x(:,end)~=0,:);
    results.day = x(x(:,end)~=0,day_index);
    tmp = diag(K_post);
    results.s2 = tmp(x(:,end)~=0,:);
    results.y = y(x(:,end)~=0,:);
    results = groupsummary(results, 'day', 'mean');
    mu = results.mean_m;
    s2 = results.mean_s2;
    days = results.day;
    counts = results.GroupCount;

end