% model for gsc synthetic data

% setup model
x_mask = [1,1,0,0,0,0];
group_mask = [0,0,0,1,0,0];
x_mean_function = {@meanMask, x_mask, {@meanLinear}}; % linear mean function
group_mean_function = {@meanMask, group_mask, {@meanConst}}; % constant group mean function
mean_function = {@meanSum, {x_mean_function, group_mean_function}};
theta.mean = [1;1;mean_mu];

% covariate covariance 

x_covariance = {@covMask, {[1,2], {@covSEard}}};
theta.cov = [log(std(x(:,1)));...        % 1
             log(std(x(:,2)));...        % 2
             log(output_scale)];         % 3

% time covariance for group trends
time_covariance = {@covMask, {3, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(length_scale); ...      % 4
             log(output_scale)];         % 5

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {4, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 6

% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};

% constant unit bias
unit_bias_covariance = {@covMask, {5, {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 7
             log(mean_sigma)];           % 8

% nonlinear unit bias
unit_error_covariance = {@covProd, {{@covMask, {3, {@covSEiso}}}, ...
                                    {@covMask, {5, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 9
             log(unit_output_scale); ... % 10
             log(0.01)];                 % 11
         
% treatment effect
treatment_effect_covariance = ...
    {@covMask, {6, {@scaled_covariance, {@scaling_function}, {@covSEiso}}}};
theta.cov = [theta.cov; ...
             T0;    ...                  % 12
             T0 + 5; ...                 % 13
             log(treat_length_scale); ...% 14
             log(treat_output_scale)];   % 15

covariance_function = {@covSum, {x_covariance, ...
                                 group_trend_covariance, ...
                                 unit_bias_covariance,   ...
                                 unit_error_covariance,  ...
                                 treatment_effect_covariance}};

% Gaussian noise
theta.lik = log(noise_scale);

% fix some hyperparameters and mildly constrain others
prior.cov  = {[], ...                               % 1:  covariate 1 length scale
              [], ...                               % 2:  covariate 2 length scale
              [], ...                               % 3:  covariate output scale
              [], ...                               % 4:  group trend length scale
              [], ...                               % 5:  group trend output scale
              {@priorSmoothBox2, -3.5, 3.5, 5}, ... % 6:  correlation
              @priorDelta, ...                      % 7:  constant unit bias
              @priorDelta, ...                      % 8:  constant unit bias
              [], ...                               % 9:  unit length scale
              [], ...                               % 10: unit output scale
              @priorDelta, ...                      % 11: unit constant 0.01
              @priorDelta, ...                      % 12: constant start of drift
              [], ...                               % 13: end of drift
              [], ...                               % 14: drift length scale
              []};                                  % 15: drift output scale
prior.lik  = {[]};                                  % 16: noise
prior.mean = {[],...                                % 17: covariate 1 mean
              [],...                                % 18: covariate 2 mean
              []};                                  % 19: group mean

inference_method = {@infPrior, @infExact, prior};