% plot sigact posterior
addpath("data");
load("data/sigact_fullbayes.mat");

chain = chain(1:30:3000,:);
FONTSIZE = 16;

clear mus;
clear s2s;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % sigact g prior
    theta_drift = theta_0;
    theta_drift.cov(9) = log(0);
    m_drift = feval(meanfunction{:}, theta_drift.mean, x);
    K_drift = feval(covfunction{:}, theta_drift.cov, x);

    % g posterior 
    [post, ~, ~] = infLaplace(theta_0, meanfunction, covfunction, likfunction, x, y);
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
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});
counterfactuals = results;
      
                
clear mus;
clear s2s;
day_index = 2;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % effect process prior
    % f+g posterior 
    [~,~,m_post,fs2] = gp(theta_0, inference_method, meanfunction, ...
                    covfunction, likfunction, x, y, x);

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
factuals = results;


fig = figure(2);
clf;

colors = ["blue","blue"];
results = factuals;
for g = 1:1
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none');
    set(h,'Facealpha', 0.2);
    hold on; plot(days, exp(mu)); 
end

BIN = 90;
XTICK = BIN*[0:1:abs(810/BIN)];
XTICKLABELS = ["Jan 2007", "Apr 2007", "Jul 2007", "Oct 2007",...
    "Jan 2008", "Apr 2008", "Jul 2008", "Oct 2008", "Jan 2009"];

set(gca, 'xtick', XTICK, ...
         'xticklabels', [],...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',12);

xlim([1, num_days]);   
legend("Factual 95% CI",...
     "Factual mean",...
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Inland (control) density");

filename = "./data/sigacttop.pdf";
set(fig, 'PaperPosition', [-2 0 22 3]); 
set(fig, 'PaperSize', [18 3]);
print(fig, filename, '-dpdf','-r300');
close;

fig = figure(2);
clf;

colors = ["blue","blue"];
results = factuals;
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none'); hold on;
    set(h,'Facealpha', 0.2);
    plot(days, exp(mu));
end

colors = ["red","green"];
results = counterfactuals;
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);
    
        
    mu = mu(treatment_day:num_days);
    s2 = s2(treatment_day:num_days);
    days = days(treatment_day:num_days);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g),'edgecolor', 'none'); hold on;
    set(h,'Facealpha', 0.2);
    plot(days, exp(mu));
end

BIN = 90;
XTICK = BIN*[0:1:abs(810/BIN)];
XTICKLABELS = ["Jan 2007", "Apr 2007", "Jul 2007", "Oct 2007",...
    "Jan 2008", "Apr 2008", "Jul 2008", "Oct 2008", "Jan 2009"];

set(gca, 'xtick', XTICK, ...
         'xticklabels', [],...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',12);
    
xlim([1, num_days]);
legend("Factual 95% CI",...
     "Factual mean",...
    "Counterfactual 95% CI","Counterfactual mean",...
    'Location', 'northwest','NumColumns',4, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Border (treated) density");

filename = "./data/sigactmid.pdf";
set(fig, 'PaperPosition', [-2 0 22 3]); 
set(fig, 'PaperSize', [18 3]);
print(fig, filename, '-dpdf','-r300');
close;