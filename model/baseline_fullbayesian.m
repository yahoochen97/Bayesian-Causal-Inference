% sampler parameters
num_chains  = 1;
num_samples = 1000;
burn_in     = 500;
jitter      = 1e-1;

num_samples = 10;
burn_in     = 5;

% setup sampler
% select index of hyperparameters to sample
theta_ind = false(size(unwrap(theta)));

theta_ind([1:3, 6:7, 10:12, 13:14, 15]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(theta_ind);

f = @(unwrapped_theta) ...
    l(unwrapped_theta, theta_ind, theta, inference_method, mean_function, ...
      covariance_function, x, y);  
  
% create and tune sampler
hmc = hmcSampler(f, theta_0 + randn(size(theta_0)) * jitter);

% tic;
% [hmc, tune_info] = ...
%     tuneSampler(hmc, ...
%                 'verbositylevel', 2, ...
%                 'numprint', 10, ...
%                 'numstepsizetuningiterations', 100, ...
%                 'numstepslimit', 500);
% toc;

use default seed for hmc sampler
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
mus = cell(size(chain,1),1);
s2s = cell(size(chain,1),1);
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [mu, s2, days, counts]=drift_posterior(theta_0, non_drift_idx,...
        mean_function, covariance_function, x, y);
    
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

filename = "./data/synthetic/fullbayes_" + HYP + "_SEED_" + SEED + ".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]);
print(fig, filename, '-dpdf','-r300');
close;

save("./data/synthetic/fullbayes_" + HYP + "_SEED_" + SEED + ".mat");

results = table(gmm_mean,sqrt(gmm_var));
results.Properties.VariableNames = {'mu','std'};
 