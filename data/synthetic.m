rng(SEED);
% initial hyperparameters
mean_mu = 0.5;
mean_sigma   = 0.01;
group_length_scale = 7;
group_output_scale = 0.1;
% unit_length_scale = 21;
unit_output_scale = 0.05;
noise_scale  = 0.01;
% rho          = 0.8;
effect       = 0.1;

HYP="rho_"+strrep(num2str(rho),'.','')+"_uls_"+num2str(unit_length_scale);

% set data size
num_days = 60;
treatment_day = 40;
num_control_units = 40;
num_treatment_units = 10;
num_units = num_control_units + num_treatment_units;

% correlated group trend
thin = 5;
x = [(1:thin:num_days)',ones(num_days/thin,1); (1:thin:num_days)',2*ones(num_days/thin,1)];

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

% plot(1:num_days, group_sample(:,1)); hold on; plot(1:num_days, group_sample(:,2));

% constant unit bias
% unit_bias = normrnd(0,mean_sigma,num_units,1);

% nonlinear unit bias
% control then treat
thin = ceil(unit_length_scale/3);
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

for i=1:num_units
    sample = mvnrnd(mu, sigma);
    
    [~,~, sample, ~] = gp(theta, @infExact, mean_function,...
    unit_covariance, @likGauss, x, sample', xs);
                
    unit_sample(i,:) = sample';
end

clear mu;
clear sigma;

% for i=25:30
%    plot(x,unit_sample(i,:)); 
%     hold on;
% end

% effect_time = (num_days - treatment_day)/2;
% effects = [zeros(1,treatment_day),...
%     effect/effect_time*(1:effect_time),...
%     effect*ones(1,num_days-treatment_day-effect_time)];

clear theta;
x = [num_days, num_days+10]';
y = [effect, effect]';
xs = (1:num_days)';
treatment_kernel = {@covSEiso};
effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};

theta.mean = 0;
theta.cov = [treatment_day; ...         
             10; ...                    
             log(30); ... 
             log(0.01)];
theta.lik = log(0);
[~,~, effects, ~] = gp(theta, @infExact, mean_function,...
    effect_covariance, @likGauss, x, y, xs);
effects = effects';

x = [repmat((1:num_days)',num_control_units,1),...
    ones(num_control_units*num_days,1),...
    reshape(repmat([1:num_control_units], num_days,1), [],1)];

x = [x; repmat((1:num_days)',num_treatment_units,1),...
    2*ones(num_treatment_units*num_days,1),...
    reshape(repmat([(num_control_units+1):num_units], num_days,1), [],1)];

% x1,x2 ~ N(0,0.5)
x1 = normrnd(0,0.5,num_units,num_days);
x2 = normrnd(0,0.5,num_units,num_days);

control = zeros(num_control_units,num_days);
treat = zeros(num_treatment_units,num_days);

for i=1:num_control_units
   control(i,:) = x1(i,:) + x2(i,:)*3 + ...
       + unit_sample(i,:) + group_sample(:,1)' ...
       + normrnd(0,noise_scale,1, num_days);
end

for i=1:num_treatment_units
   treat(i,:) = x1(i+num_control_units,:) + x2(i+num_control_units,:)*3 + ...
       group_sample(:,2)'+ unit_sample(i+num_control_units,:) + ...
       normrnd(0,noise_scale,1, num_days) + effects;
end

x = [reshape(x1',[],1),reshape(x2',[],1),x];

% group 1:control, 2:treated
y = [control; treat];
y = reshape(y',[],1);
data = array2table([x,y],'VariableNames',{'x1','x2','day','group','id','y'});
D = zeros(num_units,num_days);
D((1+num_control_units):num_units, (treatment_day+1):end) = 1;
data.D = reshape(D',[],1);

writematrix(effects(1,(treatment_day+1):end),"data/synthetic/effect_"  + HYP + "_SEED_" + SEED + ".csv");
writetable(data,"data/synthetic/data_" + HYP + "_SEED_" + SEED+ ".csv");
