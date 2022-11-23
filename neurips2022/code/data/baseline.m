function baseline(SEED, unit_length_scale, rho, effect)
% change gpml path
addpath("../gpml-matlab-v3.6-2015-07-07");
addpath("./model");
startup;

% generate synthetic data
synthetic;

% optimize hyp for synthetic data
synthetic_opt;
 
% multi gp with effect process
% writetable(results((treatment_day+1):end,:),"data/synthetic/multigp_" + HYP + "_SEED_" + SEED + ".csv");

% bayesian linear regression model
% blr;
% writetable(results((treatment_day+1):end,:),"data/synthetic/blr_" + HYP + "_SEED_" + SEED + ".csv");

% multi gp fully bayesian
baseline_fullbayesian;
writetable(results((treatment_day+1):end,:),"data/synthetic/fullbayes_" + HYP + "_SEED_" + SEED + ".csv");

% perfect correlation
% baseline_perfectcor;
% writetable(results((treatment_day+1):end,:),"data/synthetic/perfectcor_" + HYP + "_SEED_" + SEED + ".csv");

% no unit trend/group trend only
% baseline_grouptrend;
% writetable(results((treatment_day+1):end,:),"data/synthetic/grouptrend_" + HYP + "_SEED_" + SEED + ".csv");

% no group trend/unit trend only
% baseline_unittrend;
% writetable(results((treatment_day+1):end,:),"data/synthetic/unittrend_" + HYP + "_SEED_" + SEED + ".csv");

% group white noise trend (no unit trend but do not correlate group trends)
% whitenoisegroup;
% writetable(results((treatment_day+1):end,:),"data/synthetic/whitenoisegroup_" + HYP + "_SEED_" + SEED + ".csv");

% uncorrelated trends and effects
% uncorreffecttrend;
% writetable(results((treatment_day+1):end,:),"data/synthetic/uncorreffecttrend_" + HYP + "_SEED_" + SEED + ".csv");

% white noise
% whitenoise;
% writetable(results((treatment_day+1):end,:),"data/synthetic/whitenoise_" + HYP + "_SEED_" + SEED + ".csv");

% naive multi gp without effect process
% naive_cf;
% writetable(results((treatment_day+1):end,:),"data/synthetic/naivecf_" + HYP + "_SEED_" + SEED + ".csv");

% fully bayesian intrinsic coregionalization model
% naiveICM;
% writetable(results((treatment_day+1):end,:),"data/synthetic/naiveICM_" + HYP + "_SEED_" + SEED + ".csv");

% MAP intrinsic coregionalization model
% naiveICMMAP;
% writetable(results((treatment_day+1):end,:),"data/synthetic/naiveICMMAP_" + HYP + "_SEED_" + SEED + ".csv");

% no group trend/unit trend with individual hyps; individual model in Xu 2017
% baseline_unit_ITR;
% writetable(results((treatment_day+1):end,:),"data/synthetic/individual_" + HYP + "_SEED_" + SEED + ".csv");


end