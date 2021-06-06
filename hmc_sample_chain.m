function hmc_sample_chain(seed)
    addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
    addpath("model");
    % addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
    startup;
    
    load("tunesampler.mat");
    disp("mat loaded\n");
    
    % sampler parameters
    num_chains  = 5;
    num_samples = 3000;
    burn_in     = 1000;
    jitter      = 1e-1;

    % sampler parameters

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
    
    disp("finished sampling\n");

    save("results/drift" + int2str(i) + ".mat");
end
