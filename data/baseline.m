function baseline(SEED, unit_length_scale, rho)
% change gpml path
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("./model");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

synthetic;

% initial hyperparameters
mean_mu = mean(y,'all');
mean_sigma   = 0.01;
group_length_scale = 30;
group_output_scale = 0.05;
unit_length_scale = 15;
unit_output_scale = 0.05;
treat_length_scale = 15;
treat_output_scale = 0.05;
noise_scale  = 0.05;
rho          = 0.5;

% data is:
% 1: x1
% 2: x2
% 3: day number
% 4: group id
% 5: unit id
% 6: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 3)];
x(x(:, 4) == 1, end) = 0;

clear theta;
% setup model
const_mean = {@meanConst};
x_mean = {@meanMask, [true, true, false,false,false,false],  {@meanLinear}};
mean_function = {@meanSum, {const_mean, x_mean}};
theta.mean = [mean_mu;2;2];


% time covariance for group trends
time_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [log(group_length_scale); ...      % 1
             log(group_output_scale)];         % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {4, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% constant unit bias
unit_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 4
             log(mean_sigma)];           % 5

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {3, {@covSEiso}}}, ...
                                    {@covMask, {5, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 6
             log(unit_output_scale); ... % 7
             log(0.01)];                 % 8

% treatment effect
treatment_kernel = {@covSEiso};
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, treatment_kernel}}};

theta.cov = [theta.cov; ...
             treatment_day; ...          % 9
             5; ...                      % 10
             log(treat_length_scale); ...% 11
             log(treat_output_scale)];   % 12
         
% x covariance
x_covariance = {@covMask, {[1,2], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(1); ...      % 13
             log(0.01)];      % 14

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance,...
                                 x_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {[], ...                               % 1:  group trend length scale
              [], ...                               % 2:  group trend output scale
              {@priorGauss, 0.0, 1}, ...            % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5:  
              [], ...                               % 6:  unit length scale
              [], ...                               % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,1}}, ... % 10: end of drift
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,1}}, ... % 11: drift length scale
              [],...                                % 12: drift output scale
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,1}},...  % 13: x ls
              {@priorSmoothBox2, -4, -1, 5}};       % 14: x os
prior.lik  = {[]};                                  % 15: noise
prior.mean = {@priorDelta, [], []};                 % 16: mean

non_drift_idx = [2,5,7,14];

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 1;

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
fprintf("x ls: %.3f\n", exp(theta.cov(13)));
fprintf("x os: %.3f\n", exp(theta.cov(14)));
fprintf("b: %.3f\n", theta.cov(10));
          
% posterior of drift process conditioning on
% summed observation of drift + counterfactual                
theta_drift = theta.cov;
theta_drift(non_drift_idx) = log(0);
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

clear inv_V;
clear V;
clear K_drift;
clear K_post;
clear K_sum;
clear m_drift;
clear m_post;
clear m_sum;
clear tmp;

results = table(mu,sqrt(s2./counts));
results.Properties.VariableNames = {'mu','std'};

writetable(results((treatment_day+1):end,:),"data/synthetic/multigp_" + HYP + "_SEED_" + SEED + ".csv");
end