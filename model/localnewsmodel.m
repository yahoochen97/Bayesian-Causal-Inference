% localnews model

% initial hyperparameters
mean_mu           = 0.12;
mean_sigma        = 0.05;
day_sigma         = 0.01;
length_scale      = 14;
output_scale      = 0.02;
unit_length_scale = 28;
unit_output_scale = 0.02;
treat_length_scale = 30;
treat_output_scale = 0.1;
noise_scale       = 0.03;
rho               = 0.8;

mean_function = {@meanConst};
theta.mean = mean_mu;

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(length_scale); ...      % 1
             log(output_scale)];         % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {2, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% constant unit bias
unit_bias_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 4
             log(mean_sigma)];           % 5

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {1, {@covSEiso}}}, ...
                                    {@covMask, {3, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 6
             log(unit_output_scale); ... % 7
             log(0.01)];                 % 8
         
% day bias, "news happens"
day_bias_covariance = {@covMask, {4, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 9
             log(day_sigma)];            % 10

% weekday bias
weekday_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 11
             log(day_sigma)];            % 12
         

% treatment effect
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             treatment_day; ...          % 13
             7; ...                      % 14
             log(treat_length_scale); ...% 15
             log(treat_output_scale)];   % 16

covariance_function = {@covSum, {group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 day_bias_covariance,    ...
                                 weekday_bias_covariance, ...
                                 treatment_effect_covariance}};

% Gaussian noise
theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {{@priorSmoothBox2, 1.5, 4.5, 10}, ... % 1:  group trend length scale
              [], ...                               % 2:  group trend output scale
              {@priorSmoothBox2, -1.5, 1.5, 5}, ... % 3:  correlation
              @priorDelta, ...                      % 4
              @priorDelta, ...                      % 5
              {@priorSmoothBox2, 1.5, 4.5, 10}, ... % 6:  unit length scale
              [], ...                               % 7:  unit output scale
              @priorDelta, ...                      % 8
              @priorDelta, ...                      % 9
              {@priorSmoothBox2, -7, -3, 5}, ...    % 10: day effect std
              @priorDelta, ...                      % 11
              {@priorSmoothBox2, -7, -3, 5},...     % 12: weekday effect std
              @priorDelta, ...                      % 13
              {@priorGamma, 2,10}, ...              % 14: end of drift
              {@priorSmoothBox2, 1.5, 4.5, 10}, ... % 15: drift length scale
              []};                                  % 16: drift output scale
prior.lik  = {{@priorSmoothBox2, -7, -3, 5}};       % 17: noise
prior.mean = {@priorDelta};                         % 18: mean

inference_method = {@infPrior, @infExact, prior};

non_drift_idx=[2, 5, 7, 10, 12];


