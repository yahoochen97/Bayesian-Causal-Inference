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
unit_length_scale = 30;
unit_output_scale = 0.05;
noise_scale  = 0.05;
J = 5; % number of ICM samples

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

% constant unit bias
unit_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [log(0.01); ...              % 1
             log(mean_sigma)];           % 2

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {3, {@covSEiso}}}, ...
                                    {@covMask, {5, {@covDiscreteICM, num_units, J}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 3
             log(unit_output_scale); ... % 4
             normrnd(0,1,[num_units*J,1])];%5

% x covariance
x_covariance = {@covMask, {[1,2], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(1); ...      % 6
             log(0.01)];      % 7

covariance_function = {@covSum, {unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 x_covariance}};

theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {@priorDelta, ...                      % 1
              @priorDelta, ...                      % 2  
              {@priorTransform,@exp,@exp,@log,{@priorGamma,2,10}}, ... % 3:  unit length scale
              {@priorSmoothBox2, -4, -1, 5}};    % 4:  unit output scale
for i=1:num_units*J
    prior.cov{end+1} = {@priorGauss, 0, 1}; % 5: ICM
end
prior.cov{end+1}= {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}; ... % 6: x length scale
prior.cov{end+1}= {@priorSmoothBox2, -4, -1, 5};       % 7: x output scale
prior.lik  = {{@priorSmoothBox2, -4, -1, 5}};       % 8: noise std
prior.mean = {@priorDelta, [], []};                 % 9: mean

inference_method = {@infPrior, @infExact, prior};

p.method = 'LBFGS';
p.length = 10;

theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x_train, y_train);


for i=1:1
    
    theta_0 = theta;

    [~, ~, m_post, s2_post] = gp(theta_0, inference_method, mean_function,...
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
    
    mus{i} = mu;
    s2s{i} = s2./counts;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

fig = figure(1);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([days; flip(days,1)], f, [7 7 7]/8);
hold on; plot(days, gmm_mean);
plot(days, effects, "--");

filename = "./data/synthetic/naiveICMMAP_" + HYP + "_SEED_" + SEED + ".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]);
print(fig, filename, '-dpdf','-r300');
close;

save("./data/synthetic/naiveICMMAP_" + HYP + "_SEED_" + SEED + ".mat");

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};