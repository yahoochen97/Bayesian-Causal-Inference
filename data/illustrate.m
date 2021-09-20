% illustrate model = coupled group trends + unit variations + effect
rng('default');

LINEWIDTH = 4;
FONTSIZE = 20;

% initial hyperparameters
mean_mu = 0.5;
mean_sigma   = 0.01;
group_length_scale = 15;
group_output_scale = 0.05;
unit_length_scale = 7;
unit_output_scale = 0.01;
noise_scale  = 0.01;
rho          = 0.9;
effect_output_scale = 0.1;
effect_length_scale = 30;
effect       = 0.1;

% coupled group trends
num_days = 50;
treatment_day = 30;
num_control_units = 20;
num_treatment_units = 10;
num_units = num_control_units + num_treatment_units;

% correlated group trend
thin = 4;
x = [(1:thin:num_days)',ones(ceil(num_days/thin),1); (1:thin:num_days)',2*ones(ceil(num_days/thin),1)];

clear theta;
mean_function = {@meanConst};
theta.mean = mean_mu;

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(group_length_scale);  % 1
             log(group_output_scale)]; % 2

% inter-group covariance for group trends
inter_group_covariance = {@covMask, {2, {@covDiscrete2}}};
theta.cov = [theta.cov; ...
             norminv((rho + 1) / 2)];    % 3
theta.lik = log(0);
         
% complete group trend covariance
group_trend_covariance = {@covProd, {time_covariance, inter_group_covariance}};
         
mu = feval(mean_function{:},theta.mean,x);
sigma = feval(group_trend_covariance{:},theta.cov,x);

T=size(sigma,1);
group_sample = mvnrnd(mu, sigma);

xs = [(1:num_days)',ones(num_days,1); (1:num_days)',2*ones(num_days,1)];

[~,~, group_sample, ~] = gp(theta, @infExact, mean_function,...
    group_trend_covariance, @likGauss, x, group_sample', xs);

group_sample = reshape(group_sample,[],2);
group_sample(:,1) = group_sample(:,1) + 0.05;
group_sample(:,2) = group_sample(:,2) - 0.05;


fig = figure(1);
plot(1:num_days, group_sample(:,1),'blue','LineWidth',LINEWIDTH);
hold on;
plot(1:num_days, group_sample(:,2),'red','LineWidth',LINEWIDTH);


% unit variations

thin = 1;
x = (1:thin:num_days)';
xs = (1:num_days)';

clear theta;
mean_function = {@meanConst};
theta.mean = 0;

unit_covariance = {@covSEiso};
theta.cov = [log(unit_length_scale); 
             log(unit_output_scale)]; 
theta.lik = log(0);

mu = feval(mean_function{:},theta.mean,x);
sigma = feval(unit_covariance{:},theta.cov,x);
unit_sample = zeros(num_units,num_days); 

sigma = (sigma + sigma')/2;

for i=1:4
    sample = mvnrnd(mu, sigma);
    
%     [~,~, sample, ~] = gp(theta, @infExact, mean_function,...
%             unit_covariance, @likGauss, x, sample', xs);
    sample = sample - mean(sample);
    unit_sample(i,:) = sample';
end

clear mu;
clear sigma;

for i=1:1
   plot(1:num_days,(2*group_sample(:,1)'+mean(group_sample(:,1)))/3+unit_sample(i,:),'b--');
end

for i=3:3
   plot(1:num_days,(2*group_sample(:,2)'+mean(group_sample(:,2)))/3+unit_sample(i,:),'r--');
end

for i=2:2
   plot(1:num_days,(2*group_sample(:,1)'+mean(group_sample(:,1)))/3+unit_sample(i,:),'b--');
end

for i=4:4
   plot(1:num_days,(2*group_sample(:,2)'+mean(group_sample(:,2)))/3+unit_sample(i,:),'r--');
end

ylim([0.43,0.7]);

axis off;

legend('group 1', 'group 2', 'unit 1', 'unit 2','Location','northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend boxoff;

set(fig, 'PaperPosition', [0 0 5 4]); 
set(fig, 'PaperSize', [5 4]); 

filename = "data/illustrate_left.pdf";
print(fig, filename, '-dpdf','-r300');
close;

% effect 

clear theta;
treatment_kernel = {@covSEiso};
effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};
xs = linspace(0,num_days, 101)';
theta.mean = 0;
theta.cov = [treatment_day; ...         
             10; ...                    
             log(effect_length_scale); ... 
             log(effect_output_scale)];
theta.lik = log(0);

x = [num_days, num_days+10]';
y = [effect, effect]';

% mu = feval(mean_function{:},theta.mean,xs);
% sigma = feval(effect_covariance{:},theta.cov,xs,x);
% Kss = feval(effect_covariance{:},theta.cov,xs);
% 
% [post, ~, ~] = infExact(theta, mean_function, effect_covariance, {@likGauss}, x, y);
% m_post = mu + sigma*post.alpha;
% tmp = sigma*post.sW;
% K_post = Kss - tmp'*solve_chol(post.L, tmp);
% 
% effects = mvnrnd(m_post, K_post);

[~,~, effects, ~] = gp(theta, @infExact, mean_function,...
    effect_covariance, @likGauss, x, y, xs);
effects = effects';

fig = figure(1);
plot(1:num_days, 0.3*ones(1,num_days),'blue','LineWidth',LINEWIDTH);
hold on;
plot(xs, effects,'red','LineWidth',LINEWIDTH);

effects = effects(3:2:101);

ylim([-0.1,0.6]);

axis off;

legend('effect 1', 'effect 2', 'Location','northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend boxoff;

set(fig, 'PaperPosition', [0 0 5 4]); 
set(fig, 'PaperSize', [5 4]); 

filename = "data/illustrate_mid.pdf";
print(fig, filename, '-dpdf','-r300');
close;

% full model

fig = figure(1);
plot(1:num_days, group_sample(:,1),'blue','LineWidth',LINEWIDTH);
hold on;
plot(1:num_days, group_sample(:,2)+ effects'/3,'red','LineWidth',LINEWIDTH);

for i=1:1
   plot(1:num_days,(2*group_sample(:,1)'+mean(group_sample(:,1)))/3+unit_sample(i,:),'b--');
end

for i=3:3
   plot(1:num_days,(2*group_sample(:,2)'+mean(group_sample(:,2)))/3+unit_sample(i,:),'r--');
end

for i=2:2
   plot(1:num_days,(2*group_sample(:,1)'+mean(group_sample(:,1)))/3+unit_sample(i,:),'b--');
end

for i=4:4
   plot(1:num_days,(2*group_sample(:,2)'+mean(group_sample(:,2)))/3+unit_sample(i,:),'r--');
end

ylim([0.43,0.7]);

axis off;

legend('observation 1', 'observation 2','Location','northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend boxoff;

set(fig, 'PaperPosition', [0 0 5 4]); 
set(fig, 'PaperSize', [5 4]); 

filename = "data/illustrate_right.pdf";
print(fig, filename, '-dpdf','-r300');
close;