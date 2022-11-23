addpath("./model");
addpath("./data");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

SEED=1;
noise = 0.1;
var_effect = 0.25;
effect = 0.2;

effect_post = zeros(10,4);
t_post = zeros(10,4);
c_post = zeros(10,4);
tc_post = zeros(10,4);
it1=1; 
for rho = 0.09:0.1:1
% synthetic;
it2 =1;
for n = [10,50,200,1000]
sigma = eye(2,2);
sigma(1,2) = rho;
sigma(2,1) = rho;
data = mvnrnd(zeros(2,1),sigma,1);
treated = group_sample(end,1)*ones(n,1);
control = group_sample(end,2)*ones(n,1);
treated = data(:,1);
control = data(:,2);
treated_y = treated + effect+normrnd(0,noise,n,1);
control_y = control+normrnd(0,noise,n,1);

m_prior = [0;0;0];
K_prior = [[var_effect,0,0];[0,1,rho];[0,rho,1]];
K_prior = [[var_effect,0];[0,1]];

m_sum = zeros(2*n,1);
K_drift = kron(ones(1,n),[[0,var_effect];[rho,1];[1,rho]]);
K_drift = kron(ones(1,n),[[var_effect;1]]);
K_sum = kron(ones(n,n),[[1,rho];[rho,1+var_effect]])+noise^2*eye(2*n);
K_sum = kron(ones(n,n),[[1+var_effect]])+noise^2*eye(n);
K_post = K_prior - K_drift*(K_sum\K_drift');

effect_post(it1,it2) = sqrt(K_post(1,1)/n);
t_post(it1,it2) = sqrt(K_post(2,2)/n);
c_post(it1,it2) = sqrt(K_post(3,3)/n);
tc_post(it1,it2) = K_post(1,2)/sqrt(K_post(2,2))/sqrt(K_post(1,1));
it2=it2+1;
end
it1=it1+1;
end

fig=figure(1);
h = heatmap(["10","50","200","1000"], string(0.1:0.1:1), effect_post);
h.Title = 'Effect post std';
h.XLabel = 'sample size';
h.YLabel = 'correlation';

set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]); 

filename = "./results/identification_effect.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;


fig=figure(2);
h = heatmap(["10","50","200","1000"], string(0.1:0.1:1), t_post);
h.Title = 'Treat post std';
h.XLabel = 'sample size';
h.YLabel = 'correlation';
set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]); 

filename = "./results/identification_t.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;


fig = figure(3);
h = heatmap(["10","50","200","1000"], string(0.1:0.1:1), c_post);
h.Title = 'Control post std';
h.XLabel = 'sample size';
h.YLabel = 'correlation';

set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]); 

filename = "./results/identification_c.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;


fig=figure(4);
h = heatmap(["10","50","200","1000"], string(0.1:0.1:1), tc_post);
h.Title = 'Treat/effect post cor';
h.XLabel = 'sample size';
h.YLabel = 'correlation';
set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]); 

filename = "./results/identification_cor.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;

synthetic_opt;

rhos = [0.1,0.99];
TITLES = ["Low task correlation", "High task correlation"];
colors = {[166,206,227]/255, [31,120,180]/255,[178,223,138]/255};
fig=figure(1);
clf;
tiledlayout(2,2,'Padding', 'none', 'TileSpacing', 'compact');
FONTSIZE=12;
for i=1:numel(rhos)
%         h{i+numel(rhos)*(k-1)} = subplot(numel(rhos),2,i+numel(rhos)*(k-1));
nexttile;
        rho=rhos(i);
        theta.cov(3) = norminv((rhos(i) + 1) / 2);
        % f posterior 
        theta_drift = theta.cov;
        theta_drift(12) = log(0);
        m_drift = feval(mean_function{:}, theta.mean, x)*0;
        K_drift = feval(covariance_function{:}, theta_drift, x);

        theta_sum = theta.cov;
        m_sum = feval(mean_function{:}, theta.mean, x);
        K_sum = feval(covariance_function{:}, theta_sum, x);

        V = K_sum+exp(2*theta.lik)*eye(size(K_sum,1));
        inv_V = inv(V);
        m_post = m_drift + K_drift*inv_V*(y-m_sum);
        K_post = K_drift - K_drift*inv_V*K_drift;
        
        results = table;
        results.m = m_post(x(:,4)==2,:);
        results.day = x(x(:,4)==2,3);
        tmp = diag(K_post);
        results.s2 = tmp(x(:,4)==2,:);
        results.y = y(x(:,4)==2,:);
        results = groupsummary(results, 'day', 'mean');
        mu = results.mean_m;
        s2 = results.mean_s2;
        s2(1:treatment_day) = 0;
        days = results.day;
        ys = results.mean_y;
        counts = results.GroupCount;

        f = [mu+2*sqrt(s2./counts); flipdim(mu-2*sqrt(s2./counts),1)];
        fill([days; flipdim(days,1)], f, [7 7 7]/8, 'facealpha',0.5);
        hold on; plot(days,mu); 
        legend("Counterfactual 95% CI",...
     "Counterfactual mean",...
    'Location', 'southeast','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');

        title(TITLES(i), 'FontSize',FONTSIZE);
end 

for i=1:numel(rhos)
%         h{i+numel(rhos)*(k-1)} = subplot(numel(rhos),2,i+numel(rhos)*(k-1));
nexttile;
        rho=rhos(i);
        theta.cov(3) = norminv((rhos(i) + 1) / 2);
       theta_drift = theta.cov;
        theta_drift(non_drift_idx) = log(0);
        m_drift = feval(mean_function{:}, theta.mean, x)*0;
        K_drift = feval(covariance_function{:}, theta_drift, x);

        theta_sum = theta.cov;
        m_sum = feval(mean_function{:}, theta.mean, x);
        K_sum = feval(covariance_function{:}, theta_sum, x);

        V = K_sum+exp(2*theta.lik)*eye(size(K_sum,1));
        inv_V = inv(V);
        m_post = m_drift + K_drift*inv_V*(y-m_sum);
        K_post = K_drift - K_drift*inv_V*K_drift; % + exp(2*theta.lik)*eye(size(K_drift,1));

        results = table;
        results.m = m_post(x(:,end)~=0,:);
        results.day = x(x(:,end)~=0,3);
        tmp = diag(K_post);
        results.s2 = tmp(x(:,end)~=0,:);
        results.y = y(x(:,end)~=0,:);
        results = groupsummary(results, 'day', 'mean');
        mu = results.mean_m;
        s2 = results.mean_s2;
        s2(1:treatment_day) = 0;
        days = results.day;
        ys = results.mean_y;
        counts = results.GroupCount;

        f = [mu+2*sqrt(s2./counts); flipdim(mu-2*sqrt(s2./counts),1)];
        fill([days; flipdim(days,1)], f, [7 7 7]/8, 'facealpha',0.5);
        hold on; plot(days,mu); plot(days, effects, "--");
        legend("Effect 95% CI",...
         "Effect mean", "Actual effect",...
        'Location', 'southwest','NumColumns',1, 'FontSize',FONTSIZE);
        legend('boxoff');
end

set(fig, 'PaperPosition', [0 0 10 5]); 
set(fig, 'PaperSize', [10 5]); 

filename = "./results/identification.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;
