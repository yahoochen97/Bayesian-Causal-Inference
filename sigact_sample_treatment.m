% add gpml package
gpml_path = "/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07";
addpath("model");
addpath("data");
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath(gpml_path);
startup;

% set random seed
rng('default');

% load
sigacts = readtable("./data/sigacts_data.csv");
CATEGORY = ["Direct Fire"];
sigacts = sigacts(ismember(sigacts.category, CATEGORY),:);

% provinces adjacent to Pakistan
BORDER_PROVINCE = ["Nimroz",...
                  "Hilmand",...
                  "Kandahar",...
                  "Zabul",...
                  "Paktika",...
                  "Khost",...
                  "Nangarhar",...
                  "Nuristan",...
                  "Badakhshan",...
                  "Kunar"];

% set min/max date 
date_min = datetime(2007,01,01, 'Format','yyyy-MM-dd');
date_max = datetime(2008,12,31, 'Format','yyyy-MM-dd');
treatment_date = datetime(2008,8,12, 'Format','yyyy-MM-dd');
treatment_day = caldays(between(date_min,treatment_date,'days')) + 1;
num_days = caldays(between(date_min,date_max,'days')) + 1;

sigacts = sigacts(sigacts.date<=date_max & sigacts.date>=date_min, :);

sigacts.border = ismember(sigacts.province, BORDER_PROVINCE);

sigacts=groupcounts(sigacts, {'border','date','category'});

treat = zeros(num_days,numel(CATEGORY));
control = zeros(num_days,numel(CATEGORY));

for t=date_min:date_max
    i = caldays(between(date_min,t,'days')) + 1;
    for j=1:numel(CATEGORY)
        tmp = sigacts.GroupCount(sigacts.date==t & ...
            strcmp(sigacts.category, CATEGORY(j)) & ...
            sigacts.border==1);
        if numel(tmp), treat(i, j) = tmp; end
        tmp = sigacts.GroupCount(sigacts.date==t & ...
            strcmp(sigacts.category, CATEGORY(j)) & ...
             sigacts.border==0);
        if numel(tmp), control(i, j) = tmp; end
    end
end

% data is:
% 1: day number
% 2: group id: 1 for control, 2 for treat
% 3: day number (set to zero for task 1, used for drift process)
x = [(1:num_days), (1:num_days);...
    ones(1,num_days), ones(1,num_days)*2;...
    (1:num_days),(1:num_days)]';

x(x(:, 2) == 1, end) = 0;
y = [control; treat];

% init hyperparameter and define model
group_length_scale = 100;
group_output_scale = 0.5;
mean_std = 0.5;
treat_length_scale = 30;
treat_output_scale = 0.5;
rho               = 0.0;
meanfunction = {@meanMask, [false, true, false], {@meanDiscrete, 2}}; % constant mean

tmp=y(x(:,2)==1);
theta.mean = [mean(log(tmp(tmp~=0)))];
tmp=y(x(:,2)==2);
theta.mean = [theta.mean, mean(log(tmp(tmp~=0)))]; % mean of log

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(group_length_scale); ...      % 1
             log(group_output_scale)];         % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {2, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% marginalize group mean
mean_covariance = {@covMask, {2, {@covSEiso}}};

theta.cov = [theta.cov; ...
             log(0.01); ...              % 4
             log(mean_std)];             % 5

% treatment effect
treatment_effect_covariance = ...
    {@covMask, {3, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             treatment_day; ...          % 6
             30; ...                     % 7
             log(treat_length_scale); ...% 8
             log(treat_output_scale)];   % 9

covfunction = {@covSum, {group_trend_covariance, mean_covariance, treatment_effect_covariance}};

likfunction = {@likPoisson,'exp'};
theta.lik = [];

prior.cov  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,10,8}}, ... 
              [], ...
              {@priorGauss, 0.0, 1}, ... 
              @priorDelta, ...
              @priorDelta, ...
              {@priorGamma, 100, 5}, ...                     
              {@priorGamma, 3, 10}, ...             
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,8}}, ... 
              {@priorSmoothBox2, -4, -1, 5}};  
prior.lik  = {};
prior.mean = {@priorDelta, @priorDelta};


inference_method = {@infPrior, @infLaplace, prior};
non_drift_idx = [2, 5];

clear sigacts;
clear data_indirect;
clear data_direct;

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, meanfunction, ...
                    covfunction, likfunction, x, y);

% effect process prior
theta_drift = theta;
theta_drift.cov(non_drift_idx) = log(0);
m_drift = feval(meanfunction{:}, theta_drift.mean, x)*0;
K_drift = feval(covfunction{:}, theta_drift.cov, x);

% effect posterior
[post, ~, ~] = infLaplace(theta, meanfunction, covfunction, likfunction, x, y);
m_post = m_drift + K_drift*post.alpha;
tmp = K_drift.*post.sW;
K_post = K_drift - tmp'*solve_chol(post.L, tmp);

% remove control group
mu = m_post(x(:,end)~=0,:);
tmp = diag(K_post);
s2 = tmp(x(:,end)~=0,:);
days = x(x(:,end)~=0,1);

clear m_drift;
clear K_drift;
clear m_post;
clear K_post;
clear tmp;

% plot MAP
fig = figure(1);
clf;
f = [exp(mu+1.96*sqrt(s2)); exp(flip(mu-1.96*sqrt(s2),1))];
fill([days; flip(days,1)], f, [7 7 7]/8);
xlim([min(days),max(days)]);
hold on; plot(days, exp(mu));
close all;

% sampler parameters
num_chains  = 1;
num_samples = 3000;
burn_in     = 1000;
jitter      = 1e-1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([1:3, 6:9]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(theta_ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, theta_ind, theta, inference_method, meanfunction, ...
      covfunction, x, y, likfunction);  
  
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
day_index = 2;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % effect process prior
    theta_drift = theta;
    theta_drift.cov(non_drift_idx) = log(0);
    m_drift = feval(meanfunction{:}, theta_drift.mean, x)*0;
    K_drift = feval(covfunction{:}, theta_drift.cov, x);

    % effect posterior
    [post, ~, ~] = infLaplace(theta, meanfunction, covfunction, likfunction, x, y);
    m_post = m_drift + K_drift*post.alpha;
    tmp = K_drift.*post.sW;
    K_post = K_drift - tmp'*solve_chol(post.L, tmp);

    % remove control group
    mu = m_post(x(:,end)~=0,:);
    tmp = diag(K_post);
    s2 = tmp(x(:,end)~=0,:);
    days = x(x(:,end)~=0,1);

    clear m_drift;
    clear K_drift;
    clear m_post;
    clear K_post;
    clear tmp;
    
    mus{i} = mu;
    s2s{i} = s2;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

save("./data/sigact_sample_treatment" + ".mat");

fig = figure(1);
clf;
f = [exp(gmm_mean+1.96*sqrt(gmm_var)); exp(flip(gmm_mean-1.96*sqrt(gmm_var),1))];
fill([days; flip(days,1)], f, [7 7 7]/8);
hold on; plot(days, exp(gmm_mean));

filename = "./data/sigact_sample_treatment" + ".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]);
print(fig, filename, '-dpdf','-r300');
close;

