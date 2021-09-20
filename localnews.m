% add gpml package
gpml_path = "/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07";
addpath("model");
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


