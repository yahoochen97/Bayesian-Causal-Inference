% initial hyperparameters
% train data: all control + pre-treatment treated
% test data: treated
train_flag = (x(:,3)<treatment_day) | (x(:,4)==1);
test_flag = (x(:,4)==2);
x_train = x(train_flag,:);
x_test = x(test_flag,:);

y_train = y(train_flag);
y_test = y(test_flag);

mean_mu = mean(y_train,'all');
mean_sigma   = 0.01;
group_length_scale = 15;
group_output_scale = 0.05;
unit_length_scale = 30;
unit_output_scale = 0.05;
noise_scale  = 0.05;
rho          = 0.0;

% data is:
% 1: x1
% 2: x2
% 3: day number
% 4: group id
% 5: unit id

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

% x covariance
x_covariance = {@covMask, {[1,2], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(1); ...      % 9
             log(0.01)];      % 10

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
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
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ... % 9: x length scale
              {@priorSmoothBox2, -4, -1, 5}};       % 10: x output scale
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};       % 11: noise std
prior.mean = {@priorDelta, [], []};                 % 12: mean

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x_train, y_train);

fprintf("noise: %.3f\n", exp(theta.lik));
fprintf("Correlation: %.3f\n", 2*normcdf(theta.cov(3))-1);
fprintf("group ls: %.3f\n", exp(theta.cov(1)));
fprintf("group os: %.3f\n", exp(theta.cov(2)));
fprintf("unit ls: %.3f\n", exp(theta.cov(6)));
fprintf("unit os: %.3f\n", exp(theta.cov(7)));
fprintf("x ls: %.3f\n", exp(theta.cov(9)));
fprintf("x os: %.3f\n", exp(theta.cov(10)));

[~, ~, m_post, s2_post] = gp(theta, inference_method, mean_function,...
                        covariance_function, [], x_train, y_train, x_test);

results = table;
results.m = y_test - m_post;
results.day = x_test(:,3);
results.s2 = s2_post;
results = groupsummary(results, 'day', 'mean');
mu = results.mean_m;
s2 = results.mean_s2;
mu(1:treatment_day) = 0;
s2(1:treatment_day) = 0;
days = results.day;
counts = results.GroupCount;

fig=figure(3);
clf;
f = [mu+2*sqrt(s2./counts); flipdim(mu-2*sqrt(s2./counts),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
plot(days, effects, "--");

set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]); 

filename = "data/synthetic/naivecf_" + HYP + "_SEED_" + SEED + ".pdf";
print(fig, filename, '-dpdf','-r300');
close;

mu = mu(treatment_day:end);
s2 = s2(treatment_day:end);
counts = counts(treatment_day:end);

results = table(mu,sqrt(s2./counts));
results.Properties.VariableNames = {'mu','std'};