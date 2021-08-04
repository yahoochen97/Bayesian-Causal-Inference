function baseline(SEED, unit_length_scale, rho, effect)
% change gpml path
addpath("../CNNForecasting/gpml-matlab-v3.6-2015-07-07");
addpath("./model");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

% generate synthetic data
synthetic;

% % optimize hyp for synthetic data
% synthetic_opt;
% 
% % multi gp with effect process
% writetable(results((treatment_day+1):end,:),"data/synthetic/multigp_" + HYP + "_SEED_" + SEED + ".csv");
% 
% opthyp = table("noise", exp(theta.lik),...
%     "correlation",2*normcdf(theta.cov(3))-1,...
%     "group ls", exp(theta.cov(1)),...
%     "group os", exp(theta.cov(2)),...
%     "unit ls", exp(theta.cov(6)),...
%     "unit os", exp(theta.cov(7)),...
%     "effect ls", exp(theta.cov(11)),...
%     "effect os", exp(theta.cov(12)),...
%     "x ls", exp(theta.cov(13)),...
%     "x os", exp(theta.cov(14)),...
%     "b", theta.cov(10));
% writetable(opthyp,"data/synthetic/opthyp_" + HYP + "_SEED_" + SEED + ".csv");
% 
% % multi gp fully bayesian
% baseline_fullbayesian;
% writetable(results((treatment_day+1):end,:),"data/synthetic/fullbayes_" + HYP + "_SEED_" + SEED + ".csv");

% no unit trend/group trend only
% baseline_grouptrend;
% writetable(results((treatment_day+1):end,:),"data/synthetic/grouptrend_" + HYP + "_SEED_" + SEED + ".csv");

% white noise
whitenoise;
writetable(results((treatment_day+1):end,:),"data/synthetic/whitenoise_" + HYP + "_SEED_" + SEED + ".csv");

% naive multi gp without effect process
% naive_cf;
% writetable(results((treatment_day+1):end,:),"data/synthetic/naivecf_" + HYP + "_SEED_" + SEED + ".csv");

end