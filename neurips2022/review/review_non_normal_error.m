function review_non_normal_error(SEED, unit_length_scale, rho, effect, fn_name_)
% change gpml path
addpath("../code");
addpath("../code/model");
addpath("../code/data");
addpath("../code/gpml-matlab-v3.6-2015-07-07");

startup;

synthetic_fn_name = "synthetic_" + convertCharsToStrings(fn_name_);

% generate synthetic data
feval(synthetic_fn_name);

% optimize hyp for synthetic data
synthetic_opt;
writetable(results((treatment_day+1):end,:),"./results/" + fn_name_ + "_MAP_" + HYP + "_SEED_" + SEED + ".csv");

% multi gp fully bayesian
baseline_fullbayesian;
writetable(results((treatment_day+1):end,:),"./results/" + fn_name_ + "_fullbayes_" + HYP + "_SEED_" + SEED + ".csv");

end