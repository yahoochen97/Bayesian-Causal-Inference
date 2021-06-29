addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");

% addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("data");
addpath("model");
startup;

rng('default');
gscdata;


% initial hyperparameters
mean_mu           = mean(y);
mean_sigma        = 0.1;
covariate_output_scale = 0.1;
group_length_scale = 5;
group_output_scale = 0.1;
unit_length_scale = 5;
unit_output_scale = 0.1;
treat_length_scale = 5;
treat_output_scale = 1;
noise_scale       = 1;
rho               = 0.8;

gscmodel;

% data (k=2) is:
% 1: covariate 1
% 2: covariate 2
% 3: day number
% 4: group id
% 5: unit id
% 6: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 3)];
x(x(:, 4) == 1, end) = 0;

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% [~,~,fmu,fs2] = gp(theta, inference_method, mean_function, ...
%                     covariance_function, [], x, y, x);
% 
% results = table;
% results.m = fmu;
% results.day = x(:,3);
% results.s2 = fs2;
% results.y = y;
% results.group = x(:,4);
% results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

% fig = figure(2);
% clf;
% for g = 1:2
%     mu = results.mean_m(results.group==g,:);
%     s2 = results.mean_s2(results.group==g,:);
%     days = results.day(results.group==g,:);
%     ys = results.mean_y(results.group==g,:);
% 
%     f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
%     h = fill([days; flipdim(days,1)], f, [6 8 6]/8);
%     set(h,'facealpha', 0.25);
%     hold on; plot(days, mu); scatter(days, ys);
% end                

theta_drift = theta.cov;
theta_drift([3, 5, 8, 10]) = log(0);
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
results.day = x(x(:,end)~=0,3);
tmp = diag(K_post);
results.s2 = tmp(x(:,end)~=0,:);
results.y = y(x(:,end)~=0,:);
results.effect = effect(x(:,end)~=0,:);
results = groupsummary(results, 'day', 'mean');
mu = results.mean_m;
s2 = results.mean_s2;
days = results.day;
ys = results.mean_y;
effects = results.mean_effect;
counts = results.GroupCount;

fig = figure(3);
f = [mu+1.96*sqrt(s2./counts); flipdim(mu-1.96*sqrt(s2./counts),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
disp(mu(mu~=0));

plot(days, [zeros(1,T0),(1:10)]', "--");


% sampler parameters
num_chains  = 1;
num_samples = 1000;
burn_in     = 500;
jitter      = 0.1;

% setup sampler
ind = false(size(unwrap(theta)));
ind([1:3,4:5, 6, 9:10, 13:15, 16, 17:19]) = true;

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

save("results/tunegsc.mat");

% for i = 1:num_chains
%   rng(i);
%   tic;
%   [chains{i}, endpoints{i}, acceptance_ratilos(i)] = ...
%       drawSamples(hmc, ...
%                   'start', theta_0 + jitter * randn(size(theta_0)), ...
%                   'burnin', burn_in, ...
%                   'numsamples', num_samples, ...
%                   'verbositylevel', 1, ...
%                   'numprint', 10);
%   toc;
% end

% diagnostics(hmc, chains);
% samples = vertcat(chains{:});
% 
% c = exp(samples);
% c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;
% 
% figure(2);
% clf;
% plotmatrix(c);
