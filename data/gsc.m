rng('default'); % set random seed

% generate synthetic data

T = 30; % number of times
T0 = 20; % intervention time
N_tr = 5; % number of treatment units
N_co = 45; % number of control units
N = N_co + N_tr; % number of total units
k = 2; % number of covariates/factors
w = 0.8; % factor loading parameters

fs = normrnd(0,1,k,T); % time-varying factors ~ N(0,1)
xis = normrnd(0,1,T,1); % time fixed effect ~ N(0,1)
 

lambdas_tr = 2*sqrt(3)*rand(N_tr, k) - sqrt(3); % unit-specific factor loadings for control units
lambdas_co = 2*sqrt(3)*rand(N_co, k) + (1-2*w)*sqrt(3); % unit-specific factor loadings for treatment units
lambdas = [lambdas_tr; lambdas_co]; % unit-specific factor loadings

alphas_tr = 2*sqrt(3)*rand(N_tr,1) - sqrt(3); % unit fixed effect for control units
alphas_co = 2*sqrt(3)*rand(N_co,1) + (1-2*w)*sqrt(3); % unit fixed effect for treatment units
alphas = [alphas_tr; alphas_co]; % unit fixed effect

xs = zeros(N, T, k); % x_itk = 1+lambdas_i*f_t+lambdas_i1+lambdas_i2+f_1t+f_2t+e_itk

for i=1:N
   for t=1:T
      xs(i,t,:) = 1+lambdas(i,:)*fs(:,t)+sum(lambdas(i,:))+sum(fs(:,t))+ normrnd(0,1,k,1);
   end
end

Ds = zeros(N, T); % treatment indicator
Ds(1:N_tr,(T0+1):end) = 1; % 1 if t>T0,i<=N_tr else 0

deltas = zeros(N, T); % effect
deltas(1:N_tr,(T0+1):end) = repmat(1:(T-T0),N_tr,1)+normrnd(0,1,N_tr,(T-T0)); % t-T0 if t>T0,i<=N_tr else 0

ys = zeros(N, T); % y_it = delta_it*D_it+x_it1+x_it2*3+lambdas_i*f_t+alpha_i+xi_t+5+e_it

for i=1:N
   for t=1:T
      ys(i,t) = deltas(i,t)*Ds(i,t)+xs(i,t,1)+xs(i,t,2)*3 ...
        +lambdas(i,:)*fs(:,t)+alphas(t)+xis(t)+5+normrnd(0,1);
   end
end

% xs_raw = xs;
% ys_raw = ys;

x = zeros(N*T,k+3); % x1,...,xk,day,group(1 co, 2 tr),unit
y = zeros(N*T, 1);

for i=1:N
   for t=1:T
      idx=(i-1)*T+t;
      if i<=N_tr, group = 2; else, group = 1; end
      x(idx, 1:k) = xs(i,t,:);
      x(idx, k+1)=t;
      x(idx, k+2)=group;
      x(idx, k+3)=i;
      y(idx) = ys(i,t);
   end
end

% multitask GP with drift process

addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
addpath("..");
startup;

rng('default');

% initial hyperparameters
mean_mu           = 0;
mean_sigma        = 1;
length_scale      = 14;
output_scale      = 1;
unit_length_scale = 28;
unit_output_scale = 1;
treat_length_scale = 30;
treat_output_scale = 1;
noise_scale       = 1;
rho               = 0.8;

% data (k=2) is:
% 1: covariate 1
% 2: covariate 2
% 3: day number
% 4: group id
% 5: unit id
% 6: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 3)];
x(x(:, 4) == 1, end) = 0;

% make a copy before filtering
all_x = x;
all_y = y;

% discard treated data post treatment
% ind = (x(:, 1) >= treatment_day) & (x(:, 2) == 2);
% x(ind, :) = [];
% y(ind)    = [];

% skip some data during development
skip = 1;
train_ind = (randi(skip, size(y)) == 1);
x = x(train_ind, :);
y = y(train_ind);

% setup model
x_mask = [1,1,0,0,0,0];
group_mask = [0,0,0,1,0,0];
x_mean_function = {@meanMask, x_mask, {@meanLinear}}; % linear mean function
group_mean_function = {@meanMask, group_mask, {@meanConst}}; % constant group mean function
mean_function = {@meanSum, {x_mean_function, group_mean_function}};
theta.mean = [1;1;mean_mu];

% covariate covariance 

x_covariance = {@covMask, {[1,2], {@covSEard}}};
theta.cov = [log(std(x(:,1)));...        % 1
             log(std(x(:,2)));...        % 2
             log(output_scale)];         % 3

% time covariance for group trends
time_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(length_scale); ...      % 4
             log(output_scale)];         % 5

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {4, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 6

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% constant unit bias
unit_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 7
             log(mean_sigma)];           % 8

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {3, {@covSEiso}}}, ...
                                    {@covMask, {5, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 9
             log(unit_output_scale); ... % 10
             log(0.01)];                 % 11
         
% treatment effect
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             T0;    ...                  % 12
             T0 + 5; ...                 % 13
             log(treat_length_scale); ...% 14
             log(treat_output_scale)];   % 15

covariance_function = {@covSum, {x_covariance, ...
                                 group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance}};

% Gaussian noise
theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {[], ...                               % 1:  covariate 1 length scale
              [], ...                               % 2:  covariate 2 length scale
              [], ...                               % 3:  covariate output scale
              [], ...                               % 4:  group trend length scale
              [], ...                               % 5:  group trend output scale
              {@priorSmoothBox2, -3.5, 3.5, 5}, ... % 6:  correlation
              @priorDelta, ...                      % 7:  constant unit bias
              @priorDelta, ...                      % 8:  constant unit bias
              [], ...                               % 9:  unit length scale
              [], ...                               % 10: unit output scale
              @priorDelta, ...                      % 11: unit constant 0.01
              @priorDelta, ...                      % 12: constant start of drift
              [], ...                               % 13: end of drift
              [], ...                               % 14: drift length scale
              []};                                  % 15: drift output scale
prior.lik  = {[]};                                  % 16: noise
prior.mean = {[],...                                % 17: covariate 1 mean
              [],...                                % 18: covariate 2 mean
              []};                                  % 19: group mean

inference_method = {@infPrior, @infExact, prior};

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

[~,~,fmu,fs2] = gp(theta, inference_method, mean_function, ...
                    covariance_function, [], x, y, x);

results = table;
results.m = fmu;
results.day = x(:,3);
results.s2 = fs2;
results.y = y;
results.group = x(:,4);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

fig = figure(2);
clf;
for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    h = fill([days; flipdim(days,1)], f, [6 8 6]/8);
    set(h,'facealpha', 0.25);
    hold on; plot(days, mu); scatter(days, ys);
end                

theta_drift = theta.cov;
theta_drift([3, 5, 8, 10]) = log(0);
m_drift = feval(mean_function{:}, theta.mean, x)*0;
K_drift = feval(covariance_function{:}, theta_drift, x);

theta_sum = theta.cov;
% theta_sum([8]) = log(0);
m_sum = feval(mean_function{:}, theta.mean, x);
K_sum = feval(covariance_function{:}, theta_sum, x);

V = K_sum+exp(theta.lik)*eye(size(K_sum,1));
inv_V = pinv(V);
m_post = m_drift + K_drift*inv_V*(y-m_sum);
K_post = K_drift - K_drift*inv_V*K_drift;

results = table;
results.m = m_post(x(:,end)~=0,:);
results.day = x(x(:,end)~=0,3);
tmp = diag(K_post);
results.s2 = tmp(x(:,end)~=0,:);
results.y = y(x(:,end)~=0,:);
results = groupsummary(results, 'day', 'mean');
mu = results.mean_m;
s2 = results.mean_s2;
days = results.day;
ys = results.mean_y;

fig = figure(3);
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
disp(mu(mu~=0));

effect = 10;
effect_time = (T-T0);
effects = [zeros(1,T0),...
    effect/effect_time*(1:effect_time),...
    effect*ones(1,T-T0-effect_time)];

plot(days, effects, "--");


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

diagnostics(hmc, chains);
samples = vertcat(chains{:});

c = exp(samples);
c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;

figure(2);
clf;
plotmatrix(c);

                