% add path of all code

rng(SEED);
% initial hyperparameters
mean_mu = 0.5;
mean_sigma   = 0.01;
group_length_scale = 7;
group_output_scale = 0.1;
unit_output_scale = 0.001;
noise_scale  = 0.1;
effect_output_scale = 0.01;
effect_length_scale = 30;

HYP="rho_"+strrep(num2str(rho),'.','')+"_uls_"+...
    num2str(unit_length_scale) + "_effect_"+strrep(num2str(effect),'.','');

% set data size
% fewer units with 1 unit per treated group
num_days = 50;
treatment_day = 30;
num_control_units = 1;
num_treatment_units = 1;
num_units = num_control_units + num_treatment_units;

% correlated group trend
thin = 3;
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
if rho>=0.9999
    group_sample(:,1) = group_sample(:,2);
end

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

clear theta;
treatment_kernel = {@covSEiso};
effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};
xs = (1:num_days)';
theta.mean = 0;
theta.cov = [treatment_day; ...         
             10; ...                    
             log(effect_length_scale); ... 
             log(effect_output_scale)];
theta.lik = log(0);
if effect~=0
    x = [num_days, num_days+10]';
    y = [effect, effect]';
    [~,~, effects, ~] = gp(theta, @infExact, mean_function,...
        effect_covariance, @likGauss, x, y, xs);
    effects = effects';
else
    % white noise effect
    effect_length_scale = 5;
    clear theta;
    thin = 1;
    x = (1:thin:num_days)';
    mean_function = {@meanConst};
    theta.mean = 0;

    treatment_kernel = {@covSEiso};
    treatment_effect_covariance = {@scaled_covariance, {@scaling_function}, treatment_kernel};

    theta.cov = [treatment_day; ...      % 9
             10; ...                     % 10
             log(effect_length_scale); ...% 11
             log(effect_output_scale)];   % 12
    theta.lik = log(0);

    mu = feval(mean_function{:},theta.mean,x);
    sigma = feval(treatment_effect_covariance{:},theta.cov,x);
    V = feval(treatment_effect_covariance{:},theta.cov,[num_days]);
    K_int = feval(treatment_effect_covariance{:},theta.cov,x, [num_days]);
    sigma = sigma - K_int*inv(V)*K_int';

    T=size(sigma,1);
    effects = mvnrnd(mu, sigma);
end

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

% non normal error
for i=1:num_control_units
   tmp = noise_scale*normrnd(0,1,1, num_days);
   control(i,:) = x1(i,:) + x2(i,:)*3 + ...
       + unit_sample(i,:) + group_sample(:,1)' ...
       + tmp;
end

for i=1:num_treatment_units
   tmp = noise_scale*normrnd(0,1,1, num_days);
   treat(i,:) = x1(i+num_control_units,:) + x2(i+num_control_units,:)*3 + ...
       group_sample(:,2)'+ unit_sample(i+num_control_units,:) + ...
       tmp + effects;
end


x = [reshape(x1',[],1),reshape(x2',[],1),x];

% group 1:control, 2:treated
y = [control; treat];
y = reshape(y',[],1);
data = array2table([x,y],'VariableNames',{'x1','x2','day','group','id','y'});
D = zeros(num_units,num_days);
D((1+num_control_units):num_units, (treatment_day+1):end) = 1;
data.D = reshape(D',[],1);

writematrix(effects(1,(treatment_day+1):end),"./data/" + fn_name_ + "_effect_"  + HYP + "_SEED_" + SEED + ".csv");
writetable(data,"./data/" + fn_name_ + "_data_" + HYP + "_SEED_" + SEED+ ".csv");
