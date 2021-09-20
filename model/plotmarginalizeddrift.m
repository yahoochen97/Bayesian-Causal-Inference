% load one chain
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("model");
% addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

i=1;
load("results/drift" + int2str(i) + ".mat");

tic;
% thin samples
rng('default');
skip = 1;
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

toc;

fig = figure(1);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([days; flip(days,1)], f, [7 7 7]/8);
hold on; plot(days, gmm_mean);
% plot(days, effects, "--");



BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
     'xticklabels', XTICKLABELS,...
     'XTickLabelRotation',45);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'Best');
xlabel("Date"); ylabel("Effect in national news coverage");


filename = "./results/marginalizeddrift" + "_skip_" + int2str(skip) + ".pdf";
filename = "./results/localnewseffect" + "_skip_" + int2str(skip) + ".pdf";
set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]);
print(fig, filename, '-dpdf','-r300');
close;

% save("./results/marginalizeddrift" + "_skip_" + int2str(skip) + ".mat");
