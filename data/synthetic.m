% change gpml path
% addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

rng('default');

% initial hyperparameters
mean_mu = 0.5;
mean_sigma   = 0.02;
length_scale = 7;
output_scale = 0.05;
unit_length_scale = 28;
unit_output_scale = 0.02;
treat_length_scale = 14;
treat_output_scale = 0.1;
noise_scale  = 0.01;
rho          = 0.8;
effect       = 0.1;

% set data size

SEED = 1;
rng(SEED);

num_days = 50;
treatment_day = 40;
num_control_units = 10;
x = [repmat((1:num_days)',num_control_units,1),...
    ones(num_control_units*num_days,1),...
    reshape(repmat([1:num_control_units], num_days,1), [],1)];

num_treatment_units = 10;
num_units = num_control_units + num_treatment_units;
x = [x; repmat((1:num_days)',num_treatment_units,1),...
    2*ones(num_treatment_units*num_days,1),...
    reshape(repmat([(num_control_units+1):num_units], num_days,1), [],1)];

% data is:
% 1: day number
% 2: group id
% 3: unit id
% 4: day number (set to zero for task 1, used for drift process)
x = [x, x(:, 1)];
x(x(:, 2) == 1, end) = 0;

% setup model
% mean_function = {@meanMask, [false,true,false,false], {@meanDiscrete,2}};
% group mean
% theta.mean = [mean_mu-0.1, mean_mu+0.1];

mean_function = {@meanConst};
theta.mean = mean_mu;

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
    {@covMask, {4, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             treatment_day; ...          % 9
             treatment_day + 5; ...      % 10
             log(treat_length_scale); ...% 11
             log(treat_output_scale)];   % 12

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance}};

theta.lik = log(noise_scale);

mu = feval(mean_function{:},theta.mean,x);
sigma = feval(covariance_function{:},theta.cov,x)+eye(num_units*num_days)*exp(2*theta.lik);
sample = mvnrnd(mu, sigma);

control = reshape(sample(x(:,2)==1), num_days, [])';
treat = reshape(sample(x(:,2)==2), num_days, [])';

effect_time = (num_days-treatment_day)/2;
effects = [zeros(1,treatment_day),...
    effect/effect_time*(1:effect_time),...
    effect*ones(1,num_days-treatment_day-effect_time)];
treat = treat + repmat(effects,num_treatment_units,1);

fig=figure(1);
clf;
for i = 1:num_control_units % (num_control_units+1):num_units
    days = 1:num_days;
    ys = control(i,:);
    hold on; plot(days, ys);
end
title("Control units");
filename = "data/synthetic/gpcontrol_" + SEED +".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');

fig=figure(2);
clf;
for i = 1:num_treatment_units
    days = 1:num_days;
    ys = treat(i,:);
    hold on; plot(days, ys);
end
title("Treatment units");
filename = "data/synthetic/gptreat_" + int2str(SEED) +".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');

writematrix(treat,"data/synthetic/gptreat_" + SEED + ".csv");
writematrix(control,"data/synthetic/gpcontrol_" + SEED+ ".csv");