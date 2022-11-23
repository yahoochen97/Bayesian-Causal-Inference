% initial hyperparameters
mean_mu = mean(y,'all');
mean_sigma   = 0.01;
group_length_scale = 0.01;
group_output_scale = 0.05;
unit_length_scale = 30;
unit_output_scale = 0.05;
treat_length_scale = 20;
treat_output_scale = 0.05;
noise_scale  = 0.05;
rho          = 0.0;

% data is:
% 1: x1
% 2: x2
% 3: day number
% 4: group id
% 5: unit id
% 6: day number (set to zero for task 1 control, used for effect process)
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

% treatment effect
treatment_kernel = {@covSEiso};
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, treatment_kernel}}};

theta.cov = [theta.cov; ...
             treatment_day; ...          % 4
             5; ...                      % 5
             log(treat_length_scale); ...% 6
             log(treat_output_scale)];   % 7
         
% x covariance
x_covariance = {@covMask, {[1,2], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(1); ...      % 8
             log(0.01)];      % 9

covariance_function = {@covSum, {group_trend_covariance, ...
                                 treatment_effect_covariance,...
                                 x_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {@priorDelta, ... % 1:  group trend length scale
              {@priorSmoothBox2, -4, -1, 5},...     % 2:  group trend output scale
              {@priorGauss, 0.0, 1}, ...            % 3:  correlation
              @priorDelta, ...                      % 4
              {@priorGamma,10,2}, ...               % 5: full effect time
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,3}}, ... % 6: effect length scale
              {@priorSmoothBox2, -4, -1, 5}, ...    % 7: effect output scale
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ... % 8: x length scale
              {@priorSmoothBox2, -4, -1, 5}};       % 9: x output scale
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};       % 10: noise std
prior.mean = {@priorDelta, [], []};                 % 11: mean

non_drift_idx = [2,9];

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% sampler parameters
num_chains  = 1;
num_samples = 1000;
burn_in     = 500;
jitter      = 1e-1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([2:3, 5:7, 8:9, 10]) = true;

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
day_index = 3;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [mu, s2, ~, counts]=drift_posterior(theta_0, non_drift_idx,...
        mean_function, covariance_function, x, y, day_index);
    
    mus{i} = mu;
    s2s{i} = s2./counts;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

days = (1:num_days)';

fig = figure(1);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([days; flip(days,1)], f, [7 7 7]/8);
hold on; plot(days, gmm_mean);
plot(days, effects, "--");

filename = "./data/synthetic/whitenoisegroup_" + HYP + "_SEED_" + SEED + ".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]);
print(fig, filename, '-dpdf','-r300');
close;

save("./data/synthetic/whitenoisegroup_" + HYP + "_SEED_" + SEED + ".mat");

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};