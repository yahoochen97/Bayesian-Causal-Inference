function hmc_sample_chain(seed)
    addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
    % addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
    startup;
    
    load("tunesamplerdrift.mat");

    % sampler parameters
    % num_chains  = 5;
    num_samples = 1000;
    burn_in     = 500;
    jitter      = 1e-1;

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

    save("results/driftonly" + int2str(i) + ".mat");
end

% for i=1:5
%    load("results/drifthmc" + int2str(i) + ".mat");
%    chains{i} = chain;
% end
% 
% diagnostics(hmc, chains);
% samples = vertcat(chains{2});
% 
% c = exp(samples);
% c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;
% 
% figure(3);
% clf;
% plotmatrix(c);
% 
