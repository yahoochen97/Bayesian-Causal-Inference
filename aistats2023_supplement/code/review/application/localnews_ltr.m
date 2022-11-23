% change gpml path
addpath("../../code");
addpath("../../code/model");
addpath("../../code/data");
addpath("../../code/gpml-matlab-v3.6-2015-07-07");

startup;

fn_name_ = "localnews";

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

% initial hyperparameters
mean_mu           = 0.12;
mean_sigma        = 0.05;
day_sigma         = 0.01;
length_scale      = 14;
output_scale      = 0.02;
unit_length_scale = 28;
unit_output_scale = 0.05;
treat_length_scale = 30;
treat_output_scale = 0.1;
noise_scale       = 0.03;
rho               = 0.8;

mean_function = {@meanConst};
theta.mean = mean_mu;

% individual nonlinear unit bias
unit_error_covariance = {@covMask, {[4,3], {@covDiscreteIndividual, num_units}}};
theta.cov = [];

for k=1:num_units
    theta.cov = [theta.cov; 
                log(unit_length_scale); 
                log(unit_output_scale)];
end

% day bias, "news happens"
day_bias_covariance = {@covMask, {4, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 2*num_unit + 1
             log(day_sigma)];            % 2*num_unit + 2

% weekday bias
weekday_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 2*num_unit + 3
             log(day_sigma)];            % 2*num_unit + 4
         

% treatment effect
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             treatment_day; ...          % 2*num_unit + 5
             7; ...                      % 2*num_unit + 6
             log(treat_length_scale); ...% 2*num_unit + 7
             log(treat_output_scale)];   % 2*num_unit + 8

covariance_function = {@covSum, {unit_error_covariance,  ...
                                 day_bias_covariance,    ...
                                 weekday_bias_covariance, ...
                                 treatment_effect_covariance}};

% Gaussian noise
theta.lik = [log(noise_scale)];

% fix some hyperparameters and mildly constrain others
prior.cov = {};
for k=1:num_units
    prior.cov{2*k-1} = {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}};% 1:  unit trend length scale
    prior.cov{2*k} = {@priorSmoothBox2, -4, -1, 5}; % 2:  unit trend output scale
end

prior.cov{1+2*num_units}  = {@priorDelta};                      % 61
prior.cov{2+2*num_units}  = {@priorSmoothBox2, -4, -1, 5};      % 62
prior.cov{3+2*num_units}  = {@priorDelta};                      % 63
prior.cov{4+2*num_units}  = {@priorSmoothBox2, -4, -1, 5};      % 64
prior.cov{5+2*num_units}  = {@priorDelta};                      % 65
prior.cov{6+2*num_units}  = {@priorGamma, 2, 5};                % 66
prior.cov{7+2*num_units}  = {@priorTransform,@exp,@exp,@log,{@priorGamma,10,3}}; % 67
prior.cov{8+2*num_units}  = {@priorSmoothBox2, -4, -1, 5};      % 68

prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};                   % 69
prior.mean = {@priorDelta};                                     % 70

inference_method = {@infPrior, @infExact, prior};
non_drift_idx = [2:2:(2*num_units),2+2*num_units, 4+2*num_units];

p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% sampler parameters
num_chains  = 1;
num_samples = 100;
burn_in     = 50;
jitter      = 1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([1:(2*num_units), (2+2*num_units),(4+2*num_units), (6+2*num_units):(8+2*num_units), (9+2*num_units)]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(theta_ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, theta_ind, theta, inference_method, mean_function, ...
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

% use default seed for hmc sampler
rng('default');
tic;
[chain, endpoint, acceptance_ratio] = ...
  drawSamples(hmc, ...
              'start', theta_0 + jitter * randn(size(theta_0)), ...
              'burnin', burn_in, ...
              'numsamples', num_samples, ...
              'verbositylevel', 1, ...
              'numprint', 10);
toc;

% iterate all posterior samples
clear mus;
clear s2s;
day_index = 1;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [mu, s2, ~, counts]=drift_posterior(theta_0, non_drift_idx,...
        mean_function, covariance_function, x, y, day_index);

    s2 = s2 + exp(2*theta_0.lik);
    
    mus{i} = mu;
    s2s{i} = s2./counts;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};

writetable(results((63+1):end,:),"../results/" + fn_name_ + "_LTR.csv");
