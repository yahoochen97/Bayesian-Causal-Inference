% add gpml package
gpml_path = "/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07";
addpath("model");
addpath(gpml_path);
startup;

% set random seed
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

% obtain and plot drift posterior                
[fig, results] = plot_drift_posterior(theta,mean_function,...
    covariance_function, non_drift_idx, x, y);

% plot group trend of multitask gp model
plotgrouptrend;

