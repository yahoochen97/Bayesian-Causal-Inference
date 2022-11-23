function review_ltr(SEED, unit_length_scale, rho, effect, fn_name_)
% change gpml path
addpath("../code");
addpath("../code/model");
addpath("../code/data");
addpath("../code/gpml-matlab-v3.6-2015-07-07");

startup;
fn_name_ = convertCharsToStrings(fn_name_);
synthetic_fn_name = "synthetic_" + fn_name_;
disp(fn_name_);
disp(synthetic_fn_name);

% generate synthetic data
feval(synthetic_fn_name);

% initial hyperparameters
mean_mu = mean(y,'all');
mean_sigma   = 0.01;
group_length_scale = 15;
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

% individual nonlinear unit bias
unit_error_covariance = {@covMask, {[3,5], {@covDiscreteIndividual, num_units}}};
theta.cov = [];

for k=1:num_units
    theta.cov = [theta.cov; 
                log(unit_length_scale); 
                log(unit_output_scale)];
end

% treatment effect
treatment_kernel = {@covSEiso};
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, treatment_kernel}}};

theta.cov = [theta.cov; ...
             treatment_day; ...          % 61
             5; ...                      % 62
             log(treat_length_scale); ...% 63
             log(treat_output_scale)];   % 64
         
% x covariance
x_covariance = {@covMask, {[1,2], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(1); ...      % 65
             log(0.01)];      % 66

covariance_function = {@covSum, {unit_error_covariance, ...
                                 treatment_effect_covariance,...
                                 x_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov = {};
for k=1:num_units
    prior.cov{2*k-1} = {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}};% 1:  unit trend length scale
    prior.cov{2*k} = {@priorSmoothBox2, -4, -1, 5}; % 2:  unit trend output scale
end

prior.cov{1+2*num_units}  = { @priorDelta};                      % 61
prior.cov{2+2*num_units}  = {@priorGamma,10,2};                  % 62: full effect time
prior.cov{3+2*num_units}  = {@priorTransform,@exp,@exp,@log,{@priorGamma,10,3}}; % 63: effect length scale
prior.cov{4+2*num_units}  = {@priorSmoothBox2, -4, -1, 5};       % 64: effect output scale
prior.cov{5+2*num_units}  = {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}; % 65: x length scale
prior.cov{6+2*num_units}  = {@priorSmoothBox2, -4, -1, 5};       % 66: x output scale
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};         % 67: noise std
prior.mean = {@priorDelta, [], []};                   % 68 69 70: mean

non_drift_idx = [2:2:(2*num_units),6+2*num_units];

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% sampler parameters
num_chains  = 1;
num_samples = 100;
burn_in     = 50;
jitter      = 1e-1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([1:(2*num_units), (2+2*num_units):(4+2*num_units), (5+2*num_units):(6+2*num_units), (7+2*num_units)]) = true;

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

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};

writetable(results((treatment_day+1):end,:),"./results/" + fn_name_ + "_LTR_" + HYP + "_SEED_" + SEED + ".csv");

end