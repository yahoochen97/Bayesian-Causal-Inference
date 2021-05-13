function test_drift(seed)
    % addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
    addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
    startup;

    % posterior of drift process conditioning on
    % summed observation of drift + counterfactual                
    % theta_drift = theta.cov;
    % theta_drift([2, 5, 7, 10, 12]) = log(0);
    % m_drift = feval(mean_function{:}, theta.mean, x)*0;
    % K_drift = feval(covariance_function{:}, theta_drift, x);
    % 
    % theta_sum = theta.cov;
    % theta_sum([5, 10, 12]) = log(0);
    % m_sum = feval(mean_function{:}, theta.mean, x);
    % K_sum = feval(covariance_function{:}, theta_sum, x);
    % 
    % V = K_sum+exp(theta.lik)*eye(size(K_sum,1));
    % inv_V = pinv(V);
    % m_post = m_drift + K_drift*inv_V*(y-m_sum);
    % K_post = K_drift - K_drift*inv_V*K_drift;
    % 
    % results = table;
    % results.m = m_post(x(:,end)~=0,:);
    % results.day = x(x(:,end)~=0,1);
    % tmp = diag(K_post);
    % results.s2 = tmp(x(:,end)~=0,:);
    % results.y = y(x(:,end)~=0,:);
    % results = groupsummary(results, 'day', 'mean');
    % mu = results.mean_m;
    % s2 = results.mean_s2;
    % days = results.day;
    % ys = results.mean_y;
    % 
    % f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    % fill([days; flipdim(days,1)], f, [7 7 7]/8);
    % hold on; plot(days, mu);

    % sampler parameters
    % num_chains  = 5;
    num_samples = 1000;
    burn_in     = 500;
    jitter      = 1e-3;

    load("tunesampler.mat");

    i = seed;
    rng(i);
    tic;
    [chain, endpoint, acceptance_ratio] = ...
      drawSamples(hmc, ...
                  'start', theta_0 + jitter * randn(size(theta_0)), ...
                  'burnin', burn_in, ...
                  'numsamples', num_samples, ...
                  'verbositylevel', 1, ...
                  'numprint', 10);
    toc;

    save("results/drifthmc" + int2str(i) + ".mat");

% diagnostics(hmc, chains);
% samples = vertcat(chains{:});
end
