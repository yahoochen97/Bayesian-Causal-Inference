% add path
addpath("../code");
addpath("../code/model");
addpath("../code/data");
addpath("../code/gpml-matlab-v3.6-2015-07-07");
startup;

% non normal error
SEED = 1;
unit_length_scale = 21;
rho = 0.9;
effect = 0.1;

% add path of all code

rng(SEED);
% initial hyperparameters
mean_mu = 0.5;
mean_sigma   = 0.01;
group_length_scale = 7;
group_output_scale = 0.1;
unit_output_scale = 0.02;
noise_scale  = 0.1;
effect_output_scale = 0.01;
effect_length_scale = 30;

HYP="rho_"+strrep(num2str(rho),'.','')+"_uls_"+...
    num2str(unit_length_scale) + "_effect_"+strrep(num2str(effect),'.','');

% set data size
num_days = 50;
treatment_day = 30;
num_control_units = 10;
num_treatment_units = 10;
num_units = num_control_units + num_treatment_units;

% correlated group trend
thin = 3;
x = [(1:thin:num_days)',ones(ceil(num_days/thin),1); (1:thin:num_days)',2*ones(ceil(num_days/thin),1)];

clear theta;
mean_function = {@meanConst};
theta.mean = mean_mu;

% time covariance for group trends
time_covariance = {@covMask, {1, {@covMaterniso, 1}}};
theta.cov = [log(group_length_scale);  % 1
             log(group_output_scale)]; % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {2, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3
theta.lik = log(0);
         
% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};
         
mu = feval(mean_function{:},theta.mean,x);
sigma = feval(group_trend_covariance{:},theta.cov,x);

T=size(sigma,1);
group_sample = mvnrnd(mu, sigma);

xs = [(1:num_days)',ones(num_days,1); (1:num_days)',2*ones(num_days,1)];

[~,~, group_sample, ~] = gp(theta, @infExact, mean_function,...
    group_trend_covariance, @likGauss, x, group_sample', xs);

group_sample = reshape(group_sample,[],2);
if rho>=0.9999
    group_sample(:,1) = group_sample(:,2);
end

% nonlinear unit bias
% control then treat
thin = ceil(unit_length_scale/3);
x = (1:thin:num_days)';
xs = (1:num_days)';

clear theta;
mean_function = {@meanConst};
theta.mean = 0;

unit_covariance = {@covMaterniso, 1};
theta.cov = [log(unit_length_scale); 
             log(unit_output_scale)]; 
theta.lik = log(0);

mu = feval(mean_function{:},theta.mean,x);
sigma = feval(unit_covariance{:},theta.cov,x);
unit_sample = zeros(num_units,num_days); 

sigma = (sigma + sigma')/2;

for i=1:num_units
    sample = mvnrnd(mu, sigma);
    
    [~,~, sample, ~] = gp(theta, @infExact, mean_function,...
            unit_covariance, @likGauss, x, sample', xs);
                
    unit_sample(i,:) = sample';
end

clear mu;
clear sigma;

clear theta;
treatment_kernel = {@covSEiso};
effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};
xs = (1:num_days)';
theta.mean = 0;
theta.cov = [treatment_day; ...         
             10; ...                    
             log(effect_length_scale); ... 
             log(effect_output_scale)];
theta.lik = log(0);
if effect~=0
    x = [num_days, num_days+10]';
    y = [effect, effect]';
    [~,~, effects, ~] = gp(theta, @infExact, mean_function,...
        effect_covariance, @likGauss, x, y, xs);
    effects = effects';
else
    % white noise effect
    effect_length_scale = 5;
    clear theta;
    thin = 1;
    x = (1:thin:num_days)';
    mean_function = {@meanConst};
    theta.mean = 0;

    treatment_kernel = {@covSEiso};
    treatment_effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};

    theta.cov = [treatment_day; ...      % 9
             10; ...                     % 10
             log(effect_length_scale); ...% 11
             log(effect_output_scale)];   % 12
    theta.lik = log(0);

    mu = feval(mean_function{:},theta.mean,x);
    sigma = feval(treatment_effect_covariance{:},theta.cov,x);
    V = feval(treatment_effect_covariance{:},theta.cov,[num_days]);
    K_int = feval(treatment_effect_covariance{:},theta.cov,x, [num_days]);
    sigma = sigma - K_int*inv(V)*K_int';

    T=size(sigma,1);
    effects = mvnrnd(mu, sigma);
end

x = [repmat((1:num_days)',num_control_units,1),...
    ones(num_control_units*num_days,1),...
    reshape(repmat([1:num_control_units], num_days,1), [],1)];

x = [x; repmat((1:num_days)',num_treatment_units,1),...
    2*ones(num_treatment_units*num_days,1),...
    reshape(repmat([(num_control_units+1):num_units], num_days,1), [],1)];

% x1,x2 ~ N(0,0.5)
x1 = normrnd(0,0.5,num_units,num_days);
x2 = normrnd(0,0.5,num_units,num_days);

control = zeros(num_control_units,num_days);
treat = zeros(num_treatment_units,num_days);

% non normal error
% use student t error with heavier tails
% degree of freedom nu = 4
for i=1:num_control_units
   tmp = noise_scale*trnd(4,1, num_days);
   control(i,:) = x1(i,:) + x2(i,:)*3 + ...
       + unit_sample(i,:) + group_sample(:,1)' ...
       + tmp;
end

for i=1:num_treatment_units
   tmp = noise_scale*trnd(4,1, num_days);
   treat(i,:) = x1(i+num_control_units,:) + x2(i+num_control_units,:)*3 + ...
       group_sample(:,2)'+ unit_sample(i+num_control_units,:) + ...
       tmp + effects;
end

x = [reshape(x1',[],1),reshape(x2',[],1),x];

% group 1:control, 2:treated
y = [control; treat];
y = reshape(y',[],1);
data = array2table([x,y],'VariableNames',{'x1','x2','day','group','id','y'});
D = zeros(num_units,num_days);
D((1+num_control_units):num_units, (treatment_day+1):end) = 1;
data.D = reshape(D',[],1);


% define gaussian model
% initial hyperparameters
mean_mu = mean(y,'all');
mean_sigma   = 0.01;
group_length_scale = 15;
group_output_scale = 0.1;
unit_length_scale = 30;
unit_output_scale = 0.05;
treat_length_scale = 20;
treat_output_scale = 0.05;
noise_scale  = 0.1;
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
prior.cov  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ... % 1:  group trend length scale
              {@priorSmoothBox2, -4, -1, 5},...     % 2:  group trend output scale
              {@priorGauss, 0.0, 1}, ...            % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5:  
              {@priorTransform,@exp,@exp,@log,{@priorGamma,2,10}}, ... % 6:  unit length scale
              {@priorSmoothBox2, -4, -1, 5}, ...    % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              {@priorGamma,10,2}, ...               % 10: full effect time
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,3}}, ... % 11: effect length scale
              {@priorSmoothBox2, -4, -1, 5}, ...    % 12: effect output scale
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ... % 13: x length scale
              {@priorSmoothBox2, -4, -1, 5}};       % 14: x output scale
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};       % 15: noise std
prior.mean = {@priorDelta, [], []};                 % 16: mean

non_drift_idx = [2,5,7,14];

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;

theta_gauss = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

ll_gauss = gp(theta_gauss, inference_method, mean_function, ...
                    covariance_function, @likGauss, x, y);
                
% matern model
const_mean = {@meanConst};
x_mean = {@meanMask, [true, true, false,false,false,false],  {@meanLinear}};
mean_function = {@meanSum, {const_mean, x_mean}};
theta.mean = [mean_mu;2;2];


% time covariance for group trends
time_covariance = {@covMask, {3, {@covMaterniso, 1}}};
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
unit_error_covariance = {@covProd, {{@covMask, {3, {@covMaterniso, 1}}}, ...
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

theta_matern = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, @likGauss, x, y);

ll_matern = gp(theta_matern, inference_method, mean_function, ...
                    covariance_function, @likGauss, x, y);

BIC_gauss = 13*log(size(x,1)) + 2*ll_gauss;
BIC_matern = 13*log(size(x,1)) + 2*ll_matern;