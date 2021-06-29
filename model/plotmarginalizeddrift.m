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
% uppers = {};
% lowers = {};
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [mu, s2, days, counts]=drift_posterior(theta_0, non_drift_idx,...
        mean_function, covariance_function, x, y);
    
    mus{i} = mu;
    uppers{i} = mu+1.96*sqrt(s2./counts);
    lowers{i} = flip(mu-1.96*sqrt(s2./counts),1);
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(((cell2mat(uppers)- flip(cell2mat(lowers),1))/1.96/2).^2, 2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

fig = figure(1);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([days; flip(days,1)], f, [7 7 7]/8);
hold on; plot(days, gmm_mean);

toc;

filename = "./results/marginalizeddrift" + "_skip_" + int2str(skip) + ".pdf";
set(fig, 'PaperPosition', [0 0 10 10]); %Position plot at left hand corner with width 5 and height 5.
set(fig, 'PaperSize', [10 10]); %Set the paper to have width 5 and height 5.
print(fig, filename, '-dpdf','-r300');
close;

save("./results/marginalizeddrift" + "_skip_" + int2str(skip) + ".mat");

toc;