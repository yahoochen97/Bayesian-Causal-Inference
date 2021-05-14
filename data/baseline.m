% change gpml path
% addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

rng('default');

% initial hyperparameters
mean_sigma   = 0.02;
length_scale = 14;
output_scale = 0.05;
unit_length_scale = 28;
unit_output_scale = 0.02;
treat_length_scale = 30;
treat_output_scale = 0.01;
noise_scale  = 0.05;
rho          = 0.7;

% load, augment, and filter data
data_path  = "./data/synthetic";
data_idx = 2;
SEED = 1;
control = load(data_path + '/control' + int2str(data_idx) + '_' + int2str(SEED) + '.csv');
num_control_units = size(control, 1);
num_days = size(control, 2);

ind = (~isnan(control));
[this_unit, this_time] = find(ind);
x = [this_time, ones(size(this_time)), this_unit];
y = control(ind);

treat = load(data_path + '/treat' + int2str(data_idx) + '_' + int2str(SEED) + '.csv');
num_treatment_units = size(treat, 1);
num_units = num_control_units + num_treatment_units;

ind = (~isnan(treat));
[this_unit, this_time] = find(ind);
x = [x; [this_time, 2 * ones(size(this_time)), num_control_units + this_unit]];
y = [y; treat(ind)];

valid_days = unique(x(:, 1));

treatment_day = 40;

% data is:
% 1: day number
% 2: group id
% 3: unit id
% 4: day number (replicated)
% 5: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 1), x(:, 1)];
x(x(:, 2) == 1, end) = 0;

% setup model

mean_function = {@meanMask, [false, true, false, false,false], {@meanDiscrete, 2}};
% group mean
theta.mean = [mean(y(x(:,2)==1)),mean(y(x(:,2)==2))];

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

% treatment effect
treatment_effect_covariance = ...
    {@covMask, {5, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             treatment_day; ...          % 9
             treatment_day + 1; ...      % 10
             log(treat_length_scale); ...% 11
             log(treat_output_scale)];   % 12

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {[], ...                               % 1:  group trend length scale
              [], ...                               % 2:  group trend output scale
              {@priorSmoothBox2, -3.5, 3.5, 5}, ... % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5: constant mean bias prior
              [], ...                               % 6:  unit length scale
              [], ...                               % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              [], ...                               % 10: end of drift
              [], ...                               % 11: drift length scale
              []};                                  % 12: drift output scale
prior.lik  = {[]};                                  % 13: noise
prior.mean = {@priorDelta, @priorDelta};            % 14 15: mean

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;


theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% posterior of drift process conditioning on
% summed observation of drift + counterfactual                
theta_drift = theta.cov;
theta_drift([2, 5, 7]) = log(0);
m_drift = feval(mean_function{:}, theta.mean, x)*0;
K_drift = feval(covariance_function{:}, theta_drift, x);

theta_sum = theta.cov;
theta_sum([5]) = log(0);
m_sum = feval(mean_function{:}, theta.mean, x);
K_sum = feval(covariance_function{:}, theta_sum, x);

V = K_sum+exp(theta.lik)*eye(size(K_sum,1));
inv_V = pinv(V);
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

fig = figure('visible','off');
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
ylim([-0.1, 0.3]);
hold on; plot(days, mu);
fprintf("ATT: %0.3f\n", mean(mu(mu~=0)));

filename = fullfile(data_path + '/effect' + int2str(data_idx) + '_' + int2str(SEED) +".pdf");
set(fig, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');

% theta_c = theta;
% theta_c.cov = theta.cov(1:8);
% prior_c = prior;
% prior_c.cov = prior.cov(1:8);
% [~,~,fmu,fs2] = gp(theta_c, {@infPrior, @infExact, prior_c}, mean_function, ...
%                     covariance_function, [], x, y, x);

[~,~,fmu,fs2] = gp(theta, inference_method, mean_function, ...
                    covariance_function, [], x, y, x);
results = table;
results.m = fmu;
results.day = x(:,1);
results.s2 = fs2;
results.y = y;
results.group = x(:,2);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

% TODO: group
fig = figure('visible','off');
for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    fill([days; flipdim(days,1)], f, [7 7 7]/8);
    hold on; plot(days, mu); scatter(days, ys);
end
filename = fullfile(data_path + '/data' + int2str(data_idx) + '_' + int2str(SEED) +".pdf");
set(fig, 'PaperPosition', [0 0 5 5]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');


% sampler parameters
num_chains  = 5;
num_samples = 1000;
burn_in     = 500;
jitter      = 0.1;

ind = false(size(unwrap(theta)));
ind([1:3, 6:7, 10:12, 13]) = true;

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
               'numstepsizetuningiterations', 200, ...
               'numstepslimit', 500);
toc;