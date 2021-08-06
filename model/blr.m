% bayesian linear regression for two way fixed effect model
var_noise = exp(2*theta.lik);                  % var of noise level
mu_beta = theta.mean(2:3);                     % prior mean of coef for x
var_beta = exp(2*theta.cov(14))*eye(2);        % prior var of coef for x
mu_time = mean_mu*ones(num_days,1);            % prior mean of time fixed effect
var_time = exp(2*theta.cov(2))*eye(num_days);  % prior var of time fixed effect
mu_unit = 0*ones(num_units,1);                 % prior mean of unit fixed effect
var_unit = exp(2*theta.cov(7))*eye(num_units); % prior var of unit fixed effect
T_effect = num_days-treatment_day;             % num of individual effects
mu_effect = 0*ones(T_effect,1);                % prior mean of treatment effect
var_effect =exp(2*theta.cov(12))*eye(T_effect);% prior var of treatment effect

% var_time = feval(@covSEiso, theta.cov(1:2), (1:num_days)');

x_blr = x(:,1:2);
x_time = zeros(size(x,1), num_days);
x_unit = zeros(size(x,1), num_units);
x_D = zeros(size(x,1), T_effect);

for i=1:size(x,1)
   x_time(i, x(i,3)) = 1;
   x_unit(i, x(i,5)) = 1;
   if data.D(i) == 1
       x_D(i, x(i,3) - treatment_day) = 1;
   end
end

V = x_blr*var_beta*x_blr' + x_unit*var_unit*x_unit' + ...
    x_time*var_time*x_time' + x_D*var_effect*x_D' + var_noise*eye(size(x,1));
V_inv = inv(V);
K_cross = x_D*var_effect;
mu_post = mu_effect + K_cross'*V_inv*(y-x_blr*mu_beta-x_unit*mu_unit-...
    x_time*mu_time - x_D*mu_effect);
K_post = var_effect - K_cross'*V_inv*K_cross;

mu = zeros(num_days, 1);
s2 = zeros(num_days, 1);
days = (1:num_days)';
mu((treatment_day+1):num_days) = mu_post;
s2((treatment_day+1):num_days) = diag(K_post);

% fig=figure(3);
% clf;
% f = [mu+1.96*sqrt(s2); flipdim(mu-1.96*sqrt(s2),1)];
% fill([days; flipdim(days,1)], f, [7 7 7]/8);
% hold on; plot(days, mu);
% plot(days, effects, "--");
% 
% set(fig, 'PaperPosition', [0 0 10 10]); 
% set(fig, 'PaperSize', [10 10]); 
% 
% filename = "data/synthetic/blr_" + HYP + "_SEED_" + SEED + ".pdf";
% print(fig, filename, '-dpdf','-r300');
% close;

clear inv_V;
clear V;
clear K_post;
clear m_post;

results = table(mu,sqrt(s2));
results.Properties.VariableNames = {'mu','std'};