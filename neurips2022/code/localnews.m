% add gpml package
gpml_path = "./gpml-matlab-v3.6-2015-07-07";
addpath("model");
addpath("data");
addpath(gpml_path);
startup;

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

% setup model

localnewsmodel;

% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, mean_function, ...
                    covariance_function, [], x, y);

% obtain and plot drift posterior                
[fig, results] = plot_drift_posterior(theta,mean_function,...
    covariance_function, non_drift_idx, x, y);

% theta.cov([7,10,12]) = log(0);

% plot group trend of multitask gp model
[~,~,fmu,fs2] = gp(theta, inference_method, mean_function, ...
                    covariance_function, [], x, y, x);

results = table;
results.m = fmu;
results.day = x(:,1);
results.s2 = fs2;
results.y = y;
results.group = x(:,2);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

fig = figure(2);
clf;
for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [mu+1.96*sqrt(s2); flip(mu-1.96*sqrt(s2),1)];
    h = fill([days; flip(days,1)], f, [6 8 6]/8);
    set(h,'facealpha', 0.25);
    hold on; plot(days, mu); % scatter(days, ys);
end

% sampler parameters
num_chains  = 5;
num_samples = 3000;
burn_in     = 1000;
jitter      = 1e-1;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

% just sample drift parameters

% theta_ind([14:16]) = true;

theta_ind([1:3, 6, 7, 10, 12, 14:16, 17]) = true;

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

i = 1;
rng(i);
tic;
[chain, endpoint, acceptance_ratio] = ...
  drawSamples(hmc, ...
              'start', theta_0 + jitter * randn(size(theta_0)), ...
              'burnin', burn_in, ...
              'numsamples', num_samples, ...
              'verbositylevel', 1, ...
              'numprint', 10);
toc;

% thin samples
rng('default');
skip = 30;
thin_ind = (randi(skip, size(chain,1),1) == 1);
chain = chain(thin_ind,:);

% iterate all posterior samples
clear mus;
clear s2s;
day_index = 1;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [mu, s2, days, counts]=drift_posterior(theta_0, non_drift_idx,...
        mean_function, covariance_function, x, y, day_index);
    
    mus{i} = mu;
    s2s{i} = s2./counts;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;


fig = figure(1);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
h = fill([days; flip(days,1)], f, [7 7 7]/8,'edgecolor', 'none');
% set(h,'facealpha', 0.2);
hold on; plot(days, gmm_mean);

BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
         'xticklabels', XTICKLABELS,...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, num_days]);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'northwest','NumColumns',2,  'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Effect",'FontSize',FONTSIZE);

filename = "./results/localnewsbot.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
