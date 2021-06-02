% change gpml path
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("model");
% addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

rng('default');

% load, augment
load_data;

% data is:
% 1: day number
% 2: group id
% 3: unit id
% 4: day number (replicated, useful for prediction)
% 5: weekday number
% 6: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 1), mod(x(:, 1), 7), x(:, 1)];
x(x(:, 2) == 1, end) = 0;

% setup model

localnewsmodel;

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

                
% sampler parameters
num_chains  = 5;
num_samples = 5000;
burn_in     = 1000;
jitter      = 1e-1;

% setup sampler
% select index of hyperparameters to sample
ind = false(size(unwrap(theta)));

% just sample drift parameters

% ind([14:16]) = true;

ind([1:3, 6, 7, 10, 12, 14,16, 17]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, ind, theta, inference_method, mean_function, ...
      covariance_function, x, y);  
  
% create and tune sampler
hmc = hmcSampler(f, theta_0 + randn(size(theta_0)) * jitter);

tic;
[hmc, tune_info] = ...
    tuneSampler(hmc, ...
                'verbositylevel', 2, ...
                'numprint', 10, ...
                'numstepsizetuningiterations', 100, ...
                'numstepslimit', 500);
toc;
 
save("tunesampler.mat");
