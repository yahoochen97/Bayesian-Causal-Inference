% change gpml path
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
% addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
synthetic;

% initial hyperparameters
mean_mu = mean([treat',control'],'all');
mean_sigma   = 0.01;
group_length_scale = 30;
group_output_scale = 0.05;
unit_length_scale = 15;
unit_output_scale = 0.05;
treat_length_scale = 15;
treat_output_scale = 0.05;
noise_scale  = 0.05;
rho          = 0.5;

% load, augment, and filter data
ind = (~isnan(control));
[this_unit, this_time] = find(ind);
x = [this_time, ones(size(this_time)), this_unit];
y = control(ind);

ind = (~isnan(treat));
[this_unit, this_time] = find(ind);
x = [x; [this_time, 2 * ones(size(this_time)), num_control_units + this_unit]];
y = [y; treat(ind)];

% data is:
% 1: day number
% 2: group id
% 3: unit id
% 4: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 1)];
x(x(:, 2) == 1, end) = 0;

clear theta;
% setup model
mean_function = {@meanConst};
theta.mean = mean(y);

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(group_length_scale); ...      % 1
             log(group_output_scale)];         % 2

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

% treatment effect
treatment_kernel = {@covSEiso};
treatment_effect_covariance = ...
    {@covMask, {4, {@scaled_covariance, {@scaling_function}, treatment_kernel}}};

theta.cov = [theta.cov; ...
             treatment_day; ...          % 9
             5; ...                      % 10
             log(treat_length_scale); ...% 11
             log(treat_output_scale)];   % 12

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,10,1}}, ...  % 1:  group trend length scale
              {@priorSmoothBox2, -7, -3, 5}, ...    % 2:  group trend output scale
              {@priorGauss, 0.0, 1}, ...            % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5:  
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,1}}, ...  % 6:  unit length scale
              {@priorSmoothBox2, -7, -3, 5}, ...    % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              {@priorTransform,@exp,@exp,@log,{@priorGamma,5,1}}, ... % 10: end of drift
              {@priorTransform,@exp,@exp,@log,{@priorGamma,5,1}}, ...  % 11: drift length scale
              {@priorSmoothBox2, -4, -1, 5}};       % 12: drift output scale
prior.lik  = {{@priorSmoothBox2, -7, -3, 5}};       % 13: noise
prior.mean = {@priorDelta};                         % 14: mean

non_drift_idx = [2,5,7];

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

fprintf("noise: %.3f\n", exp(theta.lik));
fprintf("Correlation: %.3f\n", 2*normcdf(theta.cov(3))-1);
fprintf("group ls: %.3f\n", exp(theta.cov(1)));
fprintf("group os: %.3f\n", exp(theta.cov(2)));
fprintf("unit ls: %.3f\n", exp(theta.cov(6)));
fprintf("unit os: %.3f\n", exp(theta.cov(7)));
fprintf("effect ls: %.3f\n", exp(theta.cov(11)));
fprintf("effect os: %.3f\n", exp(theta.cov(12)));
fprintf("b: %.3f\n", theta.cov(10));
          
% posterior of drift process conditioning on
% summed observation of drift + counterfactual                
theta_drift = theta.cov;
theta_drift([2, 5, 7]) = log(0);
m_drift = feval(mean_function{:}, theta.mean, x)*0;
K_drift = feval(covariance_function{:}, theta_drift, x);

theta_sum = theta.cov;
m_sum = feval(mean_function{:}, theta.mean, x);
K_sum = feval(covariance_function{:}, theta_sum, x);

V = K_sum+exp(2*theta.lik)*eye(size(K_sum,1));
inv_V = inv(V);
m_post = m_drift + K_drift*inv_V*(y-m_sum);
K_post = K_drift - K_drift*inv_V*K_drift;

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
ys = results.mean_y;
counts = results.GroupCount;

fig=figure(3);
clf;
f = [mu+2*sqrt(s2./counts); flipdim(mu-2*sqrt(s2./counts),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
plot(days, effects, "--");

close all;
clear inv_V;
clear V;
clear K_drift;
clear K_post;
clear K_sum;
clear m_drift;
clear m_post;
clear m_sum;
clear tmp;

% filename = fullfile(data_path + '/effect_' + int2str(SEED) +".pdf");
% set(fig, 'PaperPosition', [0 0 10 10]); 
% set(fig, 'PaperSize', [10 10]); 
% print(fig, filename, '-dpdf','-r300');


% sampler parameters
num_chains  = 5;
num_samples = 1000;
burn_in     = 500;
jitter      = 0.1;

theta_ind = false(size(unwrap(theta)));
theta_ind([1:3, 6:7, 10:12, 13]) = true;


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
               'numstepsizetuningiterations', 200, ...
               'numstepslimit', 500);
toc;

save("results/tunesynthetic.mat");

% 
% for i=1:5
%     [chains{i}, endpoints{i}, acceptance_ratios{i}] = ...
%       drawSamples(hmc, ...
%                   'start', theta_0 + jitter * randn(size(theta_0)), ...
%                   'burnin', burn_in, ...
%                   'numsamples', num_samples, ...
%                   'verbositylevel', 1, ...
%                   'numprint', 10);
%     toc;
% end