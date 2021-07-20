function baseline(SEED, unit_length_scale, rho)
% change gpml path
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("./model");
% addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

% generate synthetic data
synthetic;

% optimize hyp for synthetic data
synthetic_opt;

writetable(results((treatment_day+1):end,:),"data/synthetic/multigp_" + HYP + "_SEED_" + SEED + ".csv");
end