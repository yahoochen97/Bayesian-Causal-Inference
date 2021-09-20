% plot sigact posterior
addpath("data");
load("data/sigact_fullbayes.mat");

% chain = chain(1:1000:3000,:);

clear mus;
clear s2s;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % sigact g prior
    theta_drift = theta;
    theta_drift.cov(9) = log(0);
    m_drift = feval(meanfunction{:}, theta_drift.mean, x);
    K_drift = feval(covfunction{:}, theta_drift.cov, x);

    % g posterior 
    [post, ~, ~] = infLaplace(theta, meanfunction, covfunction, likfunction, x, y);
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

colors = ["green","yellow"];
for g = 2:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g));
    set(h,'facealpha', 0.2);
    hold on; plot(days, exp(mu)); 
end
                
                
clear mus;
clear s2s;
day_index = 2;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    % effect process prior
    % f+g posterior 
    [~,~,m_post,fs2] = gp(theta, inference_method, meanfunction, ...
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

colors = ["red","blue"];
for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, colors(g)); hold on;
    set(h,'facealpha', 0.2);
    plot(days, exp(mu));
end

BIN = 90;
XTICK = BIN*[0:1:abs(810/BIN)];
XTICKLABELS = ["Jan 2007", "Apr 2007", "Jul 2007", "Oct 2007",...
    "Jan 2008", "Apr 2008", "Jul 2008", "Oct 2008", "Jan 2009"];

set(gca, 'xtick', XTICK, ...
     'xticklabels', XTICKLABELS,...
     'XTickLabelRotation',45);
    
legend("Border counterfactual 95% CI",...
     "Border counterfactual mean",...
    "Inland factual 95% CI","Inland factual mean",...
    "Border factual 95% CI","Border factual mean",...
    'Location', 'Best');
xlabel("Date"); ylabel("Direct fire density");

filename = "./data/sigact.pdf";
set(fig, 'PaperPosition', [0 0 10 10]); 
set(fig, 'PaperSize', [10 10]);
print(fig, filename, '-dpdf','-r300');
close;