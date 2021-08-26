% plot sigact posterior
addpath("data");
load("data/sigact_fullbayes.mat");

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
for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, [6 8 6]/8);
    set(h,'facealpha', 0.25);
    hold on; plot(days, exp(mu)); 
end

% f+g posterior 
[~,~,fmu,fs2] = gp(theta, inference_method, meanfunction, ...
                    covfunction, likfunction, x, y, x);

results = table;
results.m = fmu;
results.day = x(:,1);
results.s2 = fs2;
results.y = y;
results.group = x(:,2);
results = groupsummary(results, {'day','group'}, 'mean',{'m','s2', 'y'});

for g = 1:2
    mu = results.mean_m(results.group==g,:);
    s2 = results.mean_s2(results.group==g,:);
    days = results.day(results.group==g,:);
    ys = results.mean_y(results.group==g,:);

    f = [exp(mu+2*sqrt(s2)); exp(flip(mu-2*sqrt(s2),1))];
    h = fill([days; flip(days,1)], f, [6 8 6]/8); hold on;
    set(h,'facealpha', 0.75);
    plot(days, exp(mu));  scatter(days, ys);
end
