addpath("model");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

likfunction = {@likGauss};
load("results/drift1.mat");

FONTSIZE = 16;

% xs = zeros(num_days*2*num_units,3);
% for i=1:num_days
%    for j=1:num_units
%        xs()
%    end
% end

% plot group trend of multitask gp model

% thin samples
rng('default');
skip = 30;
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
    theta_drift.cov([5,7]) = log(0);
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

fmu = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_mean = fmu;
fs2 = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

% plot posterior g
results = table;
results.m = fmu;
results.day = x(:,1);
results.s2 = fs2;
results.y = y;
results.group = x(:,2);
results.unit = x(:,3);
counterfactuals = results;
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

counterfactuals = groupsummary(counterfactuals, {'day','group'}, 'mean',{'m','s2', 'y'});
                
clear mus;
clear s2s;
day_index = 2;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);


    % f+g posterior 
%     [~,~,m_post,fs2] = gp(theta_0, inference_method, mean_function, ...
%                     covariance_function, likfunction, x, y, x);

    % localnews g prior
    theta_drift = theta_0;
%     theta_drift.cov(16) = log(0);
    
    % remove day/weekday
    theta_drift.cov([10,12]) = log(0);
    theta_drift.cov([5,7]) = log(0);
    m_drift = feval(mean_function{:}, theta_drift.mean, x);
    K_drift = feval(covariance_function{:}, theta_drift.cov, x);

%     % g posterior 
    [post, ~, ~] = infExact(theta_0, mean_function, covariance_function, likfunction, x, y);
    m_post = m_drift + K_drift*post.alpha;
    tmp = K_drift.*post.sW;
    K_post = K_drift - tmp'*solve_chol(post.L, tmp);


    % remove control group
    mus{i} = m_post;
    s2s{i} = diag(K_post);
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
results.unit = x(:,3);
factuals = results;
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});
factuals = groupsummary(factuals, {'day','group'}, 'mean',{'m','s2', 'y'});

fig = figure(2);
clf;

colors = ["blue"];
results = factuals;
for g = 1:1
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);
    
    mu = mu(2:end);
    s2 = s2(2:end);
    days = days(2:end);
    ys = ys(2:end);

    f = [(mu+1.96*sqrt(s2)); (flip(mu-1.96*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none');
    set(h,'facealpha', 0.2);
    hold on; plot(days, (mu)); 
end

BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
         'xticklabels', [],...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, num_days]);
ylim([0.05,0.25]);
    
legend("Factual 95% CI",...
     "Factual mean",...
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("News coverage (control)",'FontSize',FONTSIZE);

filename = "./results/localnewstop.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;

fig = figure(2);
clf;

colors = ["black","blue"];
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);
   
    mu = mu(2:end);
    s2 = s2(2:end);
    days = days(2:end);
    ys = ys(2:end);

    f = [(mu+1.96*sqrt(s2)); (flip(mu-1.96*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none'); hold on;
    set(h,'facealpha', 0.2);
    plot(days, (mu));
end

colors=["black","green"];
results = counterfactuals;
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);
    
    mu = mu(days>=treatment_day);
    s2 = s2(days>=treatment_day);
    ys = ys(days>=treatment_day);
    days = days(days>=treatment_day);

    f = [(mu+1.96*sqrt(s2)); (flip(mu-1.96*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none'); hold on;
    set(h,'facealpha', 0.2);
    plot(days, (mu));
end


BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
         'xticklabels', [],...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, num_days]);
    
legend("Factual 95% CI",...
     "Factual mean",...
    "Counteractual 95% CI","Counterfactual mean",...
    'Location', 'northwest', 'NumColumns',4, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("News coverage (treated)",'FontSize',FONTSIZE);

filename = "./results/localnewsmid.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
