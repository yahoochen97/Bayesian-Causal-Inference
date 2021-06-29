rng('default'); % set random seed

% generate synthetic data

T = 30; % number of times
T0 = 20; % intervention time
N_tr = 5; % number of treatment units
N_co = 45; % number of control units
N = N_co + N_tr; % number of total units
k = 2; % number of covariates/factors
w = 0.8; % factor loading parameters
noise = 1; % noise std


fs = normrnd(0,1,k,T); % time-varying factors ~ N(0,1)
xis = normrnd(0,1,T,1); % time fixed effect ~ N(0,1)
 

lambdas_tr = 2*sqrt(3)*rand(N_tr, k) - sqrt(3); % unit-specific factor loadings for control units
lambdas_co = 2*sqrt(3)*rand(N_co, k) + (1-2*w)*sqrt(3); % unit-specific factor loadings for treatment units
lambdas = [lambdas_tr; lambdas_co]; % unit-specific factor loadings

alphas_tr = 2*sqrt(3)*rand(N_tr,1) - sqrt(3); % unit fixed effect for control units
alphas_co = 2*sqrt(3)*rand(N_co,1) + (1-2*w)*sqrt(3); % unit fixed effect for treatment units
alphas = [alphas_tr; alphas_co]; % unit fixed effect

xs = zeros(N, T, k); % x_itk = 1+lambdas_i*f_t+lambdas_i1+lambdas_i2+f_1t+f_2t+e_itk

for i=1:N
   for t=1:T
      xs(i,t,:) = 1+lambdas(i,:)*fs(:,t)+sum(lambdas(i,:))+sum(fs(:,t))+ noise*normrnd(0,1,k,1);
   end
end

Ds = zeros(N, T); % treatment indicator
Ds(1:N_tr,(T0+1):end) = 1; % 1 if t>T0,i<=N_tr else 0

deltas = zeros(N, T); % effect
deltas(1:N_tr,(T0+1):end) = repmat(1:(T-T0),N_tr,1)+noise*normrnd(0,1,N_tr,(T-T0)); % t-T0 if t>T0,i<=N_tr else 0

ys = zeros(N, T); % y_it = delta_it*D_it+x_it1+x_it2*3+lambdas_i*f_t+alpha_i+xi_t+5+e_it

for i=1:N
   for t=1:T
      ys(i,t) = deltas(i,t)*Ds(i,t)+xs(i,t,1)+xs(i,t,2)*3 ...
        +lambdas(i,:)*fs(:,t)+alphas(t)+xis(t)+5+noise*normrnd(0,1);
   end
end

x = zeros(N*T,k+3); % x1,...,xk,day,group(1 co, 2 tr),unit
y = zeros(N*T, 1);
D = zeros(N*T, 1);
effect = zeros(N*T, 1);

for i=1:N
   for t=1:T
      idx=(i-1)*T+t;
      if i<=N_tr, group = 2; else, group = 1; end
      x(idx, 1:k) = xs(i,t,:);
      x(idx, k+1)=t;
      x(idx, k+2)=group;
      x(idx, k+3)=i;
      y(idx) = ys(i,t);
      D(idx) = Ds(i,t);
      effect(idx) = deltas(i,t);
   end
end

% writematrix([x,y,D,effect],'./data/synthetic/gsc.csv');

                