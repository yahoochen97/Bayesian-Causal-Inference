addpath("model");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

likfunction = {@likGauss};
load("results/drift1.mat");

% plot group trend of multitask gp model

% thin samples
rng('default');
skip = 100;
thin_ind = (randi(skip, size(chain,1),1) == 1);
chain = chain(thin_ind,:);

% [~,~,fmu,fs2] = gp(theta, inference_method, mean_function, ...
%                     covariance_function, [], x, y, x);
% 
% results = table;
% results.m = fmu;
% results.day = x(:,1);
% results.s2 = fs2;
% results.y = y;
% results.group = x(:,2);
% results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

% fig = figure(2);
% clf;
% for g = 1:2
%     mu = results.mean_m(results.group==g,:);
%     s2 = results.mean_s2(results.group==g,:);
%     days = results.day(results.group==g,:);
%     ys = results.mean_y(results.group==g,:);
% 
%     f = [mu+1.96*sqrt(s2); flip(mu-1.96*sqrt(s2),1)];
%     h = fill([days; flip(days,1)], f, [6 8 6]/8);
%     set(h,'facealpha', 0.25);
%     hold on; plot(days, mu); scatter(days, ys);
% end

clear mus;
clear s2s;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % localnews g prior
    theta_drift = theta_0;
    theta_drift.cov(16) = log(0);
    
    % remove day/weekday
    theta_drift.cov([10,12]) = log(0);
    m_drift = feval(mean_function{:}, theta_drift.mean, x);
    K_drift = feval(covariance_function{:}, theta_drift.cov, x);

    % g posterior 
    [post, ~, ~] = infExact(theta_0, mean_function, covariance_function, likfunction, x, y);
    m_post = m_drift + K_drift*post.alpha;
    tmp = K_drift.*post.sW;
    K_post = K_drift - tmp'*solve_chol(post.L, tmp);

    % remove control group
    mus{i} = m_post;
    s2s{i} = diag(K_post);
end

% plot posterior g
results = table;
results.m = m_post;
results.day = x(:,1);
results.s2 = diag(K_post);
results.y = y;
results.group = x(:,2);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});
fig = figure(2);
clf;

colors = ["yellow","green"];
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [(mu+1.96*sqrt(s2)); (flip(mu-1.96*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g));
    set(h,'facealpha', 0.15);
    hold on; plot(days, (mu)); 
end
                
                
clear mus;
clear s2s;
day_index = 2;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % f+g posterior 
%     [~,~,m_post,fs2] = gp(theta, inference_method, mean_function, ...
%                     covariance_function, likfunction, x, y, x);

% localnews g prior
    theta_drift = theta_0;
%     theta_drift.cov(16) = log(0);
    
    % remove day/weekday
    theta_drift.cov([10,12]) = log(0);
    m_drift = feval(mean_function{:}, theta_drift.mean, x);
    K_drift = feval(covariance_function{:}, theta_drift.cov, x);

    % g posterior 
    [post, ~, ~] = infExact(theta_0, mean_function, covariance_function, likfunction, x, y);
    m_post = m_drift + K_drift*post.alpha;
    tmp = K_drift.*post.sW;
    K_post = K_drift - tmp'*solve_chol(post.L, tmp);


    % remove control group
    mus{i} = m_post;
    s2s{i} = fs2;
end

fmu = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_mean = fmu;
fs2 = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

results = table;
results.m = fmu;
results.day = x(:,1);
results.s2 = fs2;
results.y = y;
results.group = x(:,2);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

colors = ["red","blue"];
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [(mu+1.96*sqrt(s2)); (flip(mu-1.96*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g)); hold on;
    set(h,'facealpha', 0.15);
    plot(days, (mu));
end

BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
     'xticklabels', XTICKLABELS,...
     'XTickLabelRotation',45);
    
legend("Acquired counterfactual 95% CI",...
     "Acquired counterfactual mean",...
    "Acquired factual 95% CI","Acquired factual mean",...
    'Location', 'Best');
xlabel("Date"); ylabel("Proportion of localnews coverage.");

filename = "./results/localnewsgrouptrend.pdf";
set(fig, 'PaperPosition', [0 0 20 10]); 
set(fig, 'PaperSize', [20 10]);
print(fig, filename, '-dpdf','-r300');
close;
