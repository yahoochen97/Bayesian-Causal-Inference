function test_drift(seed)
    addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
    % addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
    startup;

    % sampler parameters
    % num_chains  = 5;
    num_samples = 1000;
    burn_in     = 500;
    jitter      = 0.1;

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

% c = exp(chain);
% c(:, 3) = 2 * normcdf(chain(:, 3)) - 1;
% 
% figure(3);
% clf;
% plotmatrix(c);

end
