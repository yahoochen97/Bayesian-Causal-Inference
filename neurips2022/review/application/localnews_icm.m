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
% train data: all control + pre-treatment treated
% test data: treated
train_flag = (x(:,4)<=treatment_day) | (x(:,2)==1);
test_flag = (x(:,2)==2);
x_train = x(train_flag,:);
x_test = x(test_flag,:);

y_train = y(train_flag);
y_test = y(test_flag);

mean_mu = mean(y_train,'all');
day_sigma = 0.01;
mean_sigma   = 0.01;
unit_length_scale = 30;
unit_output_scale = 0.05;
noise_scale  = 0.05;
J = 5; % number of ICM samples

clear theta;
% setup model
mean_function = {@meanConst};
theta.mean = mean_mu;

% constant unit bias
unit_bias_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [log(0.01); ...              % 1
             log(mean_sigma)];           % 2

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {1, {@covSEiso}}}, ...
                                    {@covMask, {3, {@covDiscreteICM, num_units, J}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 3
             log(1); ... % 4
             unit_output_scale*normrnd(0,1,[num_units*J,1])];%5

% day bias, "news happens"
day_bias_covariance = {@covMask, {4, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 6
             log(day_sigma)];            % 7

% weekday bias
weekday_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 8
             log(day_sigma)];            % 9
         

covariance_function = {@covSum, {day_bias_covariance, ...
                                weekday_bias_covariance,...
                                unit_bias_covariance,   ...
                                 unit_error_covariance}};

theta.lik = [log(noise_scale)];

% fix some hyperparameters and mildly constrain others
prior.cov  = {@priorDelta, ...                      % 1
              @priorDelta, ...                      % 2  
              {@priorTransform,@exp,@exp,@log,{@priorGamma,2,5}}, ... % 3:  unit length scale
              @priorDelta};    % 4:  unit output scale
for i=1:num_units*J
    prior.cov{end+1} = {@priorGauss, 0, unit_output_scale^2}; % 5: ICM
end
prior.cov{end+1}= {@priorDelta}; ... % 6:
prior.cov{end+1}= {@priorSmoothBox2, -4, -1, 5};       % 7: 
prior.cov{end+1}= {@priorDelta}; ... % 8:
prior.cov{end+1}= {@priorSmoothBox2, -4, -1, 5};       % 9: 
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};       % 10: noise std
prior.mean = {@priorDelta};                 % 11: mean

inference_method = {@infPrior, @infExact, prior};
 
p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x_train, y_train);

% sampler parameters
num_chains  = 1;
num_samples = 100;
burn_in     = 50;
% use a larger jitter
jitter      = 1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([3, 5:(4+num_units*J),6+num_units*J,8+num_units*J, 9+num_units*J]) = true;
% marginalize x mean
% theta_ind(end)= true;
% theta_ind(end-1)=true;

theta_0 = unwrap(theta);
theta_0 = theta_0(theta_ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, theta_ind, theta, inference_method, mean_function, ...
      covariance_function, x_train, y_train);
  
% create and tune sampler
hmc = hmcSampler(f, theta_0 + randn(size(theta_0)) * jitter);

tic;
[hmc, tune_info] = ...
    tuneSampler(hmc, ...
                'verbositylevel', 2, ...
                'numprint', 1, ...
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

    [~, ~, m_post, s2_post] = gp(theta_0, inference_method, mean_function,...
                        covariance_function, [], x_train, y_train, x_test);

    results = table;
    results.m = y_test - m_post;
    results.day = x_test(:,1);
    results.s2 = s2_post;
    results = groupsummary(results, 'day', 'mean');
    mu = results.mean_m;
    s2 = results.mean_s2;
    days = results.day;
    counts = results.GroupCount;
    
    mus{i} = mu;
    s2s{i} = s2./counts;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};

writetable(results((63+1):end,:),"../results/" + fn_name_ + "_ICM.csv");
