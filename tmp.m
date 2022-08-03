% add gpml package
gpml_path = "/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07";
addpath("model");
addpath("data");
addpath(gpml_path);
startup;

% set random seed
rng('default');

% load
data = readtable("./tmp.csv");
data = data(2:52,:);
[x,order] = sort(data.Var1);
y = data.Var2(order);

length_scale = 1;
output_scale = 1;
meanfunction = {@meanConst}; % constant mean
theta.mean = 0;

covfunction = {@covSEiso};
theta.cov = [log(length_scale); ...      % 1
             log(output_scale)];         % 2

likfunction = {@likGauss};
theta.lik = [log(0.2)];

prior.cov  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,1,0.5}}, ...             
              {@priorTransform,@exp,@exp,@log,{@priorGamma,1,0.5}}};  
prior.lik  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,1,5}}};
prior.mean = {@priorDelta};

inference_method = {@infPrior, @infExact, prior};


% find MAP
p.method = 'LBFGS';
p.length = 100;
theta = minimize_v2(theta, @gp, p, inference_method, meanfunction, ...
                    covfunction, likfunction, x, y);

xs = linspace(-3,3,100)';
[~,~, mu, s2] = gp(theta, inference_method, meanfunction, covfunction, [], x, y, xs);
                
% plot MAP
fig = figure(1);
clf;
f = [mu+1.96*sqrt(s2);flip(mu-1.96*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8);
xlim([min(xs),max(xs)]);
hold on; scatter(x,y,'*'); plot(xs, mu); title("Ryan example MAP");


% sampler parameters
num_chains  = 1;
num_samples = 100;
burn_in     = 200;
jitter      = 1e-1;

theta_ind = false(size(unwrap(theta)));

theta_ind([1:3]) = true;

theta_0 = unwrap(theta);
theta_0 = theta_0(theta_ind);

f = @(unwrapped_theta) customize_ll(unwrapped_theta, theta_ind, theta, inference_method, meanfunction, ...
      covfunction, x, y);  
  
% create and tune sampler
hmc = hmcSampler(f, theta_0 + randn(size(theta_0)) * jitter);

tic;
[hmc, tune_info] = ...
    tuneSampler(hmc, ...
                'verbositylevel', 2, ...
                'numprint', 10, ...
                'numstepsizetuningiterations', 100, ...
                'numstepslimit', 500);
toc;

% use default seed for hmc sampler
rng('default');
tic;
[chain, endpoint, acceptance_ratio] = ...
  drawSamples(hmc, ...
              'start', theta_0 + jitter * randn(size(theta_0)), ...
              'burnin', burn_in, ...
              'numsamples', num_samples, ...
              'verbositylevel', 1, ...
              'numprint', 10);
toc;

clear mus;
clear s2s;
for i=1:size(chain,1)
    
    theta_0 = unwrap(theta);
    theta_0(theta_ind)=chain(i,:);
    theta_0 = rewrap(theta, theta_0);

    [~, ~, mu, s2] = gp(theta_0, inference_method, meanfunction,...
      covfunction, [], x, y, xs);
    
    mus{i} = mu;
    s2s{i} = s2;
end

gmm_mean = mean(cell2mat(mus),2);
gmm_s2 = mean(cell2mat(s2s),2);
gmm_var = gmm_s2 + mean(cell2mat(mus).^2,2) - gmm_mean.^2;

fig = figure(2);
clf;
f = [gmm_mean+1.96*sqrt(gmm_var); flip(gmm_mean-1.96*sqrt(gmm_var),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8, 'edgecolor', 'none');
hold on; plot(xs, gmm_mean);
scatter(x,y,'*'); title("Ryan example Marginalized");

fig = figure(3);
expchain = exp(chain);
plotmatrix(expchain);

function [f, g] = ...
  customize_ll(unwrapped_theta, ind, theta, inference_method, mean_function, covariance_function, x, y, likfunction)
  
  if nargin==8, likfunction = []; end
  t = unwrap(theta);
  t(ind) = unwrapped_theta;
  theta = rewrap(theta, t);

  [f, g] = gp(theta, inference_method, mean_function,...
      covariance_function, likfunction, x, y);
  g = unwrap(g);
  f = -f;
  g = -g(ind);

end

