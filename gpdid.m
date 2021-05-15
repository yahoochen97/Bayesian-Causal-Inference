addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

load("results/hmc.mat");
rng('default');

% initial hyperparameters
mean_mu           = 0.12;
mean_sigma        = 0.05;
day_sigma         = 0.01;
length_scale      = 14;
output_scale      = 0.02;
unit_length_scale = 28;
unit_output_scale = 0.02;
noise_scale       = 0.03;
rho               = 0.8;

% load, augment, and filter data
load_data;

% data is:
% 1: day number
% 2: group id
% 3: unit id
% 4: day number (replicated, useful for prediction)
% 5: weekday number
x = [x, x(:, 1), mod(x(:, 1), 7)];

% make a copy before filtering
all_x = x;
all_y = y;

% discard treated data post treatment
ind = (x(:, 1) >= treatment_day) & (x(:, 2) == 2);
x(ind, :) = [];
y(ind)    = [];

% skip some data during development
skip = 1;
train_ind = (randi(skip, size(y)) == 1);
x = x(train_ind, :);
y = y(train_ind);

% setup model

mean_function = {@meanConst};
theta.mean = mean_mu;

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(length_scale); ...      % 1
             log(output_scale)];         % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {2, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% constant unit bias
unit_bias_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 4
             log(mean_sigma)];           % 5

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {1, {@covSEiso}}}, ...
                                    {@covMask, {3, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 6
             log(unit_output_scale); ... % 7
             log(0.01)];                 % 8
         
% day bias, "news happens"
day_bias_covariance = {@covMask, {4, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 9
             log(day_sigma)];            % 10

% weekday bias
weekday_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 11
             log(day_sigma)];            % 12

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 day_bias_covariance,    ...
                                 weekday_bias_covariance}};

% Gaussian noise
theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {[], ...                               % 1:  group trend length scale
              [], ...                               % 2:  group trend output scale
              {@priorSmoothBox2, -3.5, 3.5, 5}, ... % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5
              [], ...                               % 6:  unit length scale
              [], ...                               % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              {@priorSmoothBox2, -9, -3, 5}, ...    % 10: weekday effect std
              @priorDelta, ...                      % 11
              {@priorSmoothBox2, -9, -3, 5}};       % 12: weekday effect std
prior.lik  = {{@priorSmoothBox2, -9, -3, 5}};       % 13: noise
prior.mean = {@priorDelta};                         % 14: mean

inference_method = {@infPrior, @infExact, prior};

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% make_intermediate_plot_unitw;

% sampler parameters
num_chains  = 5;
num_samples = 50;
burn_in     = 50;
jitter      = 0.1;

% setup sampler
ind = false(size(unwrap(theta)));
ind([1:3, 6, 7, 10, 12, end - 1]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, ind, theta, inference_method, mean_function, ...
      covariance_function, x, y);

% create and tune sampler
hmc = hmcSampler(f, theta_0);
% tic;
% [hmc, tune_info] = ...
%    tuneSampler(hmc, ...
%                'verbositylevel', 2, ...
%                'numprint', 100, ...
%                'numstepsizetuningiterations', 100, ...
%                'numstepslimit', 500);
% toc;


% run several chains
for i = 1:num_chains
  rng(i);
  tic;
  [chains{i}, endpoints{i}, acceptance_ratilos(i)] = ...
      drawSamples(hmc, ...
                  'start', theta_0 + jitter * randn(size(theta_0)), ...
                  'burnin', burn_in, ...
                  'numsamples', num_samples, ...
                  'verbositylevel', 1, ...
                  'numprint', 10);
  toc;
end

save("results/hmc.mat");

diagnostics(hmc, chains);
samples = vertcat(chains{:});

c = exp(samples);
c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;

figure(2);
clf;
plotmatrix(c);

% make_final_plot_unitw;


