% inf laplace

n=1000;
x = linspace(-3, 3, n)';                 % 100 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, n, 1);% 100 noisy training targets 

meanfunc = {@meanZero};             % zero mean function
covfunc = {@covSEiso};              % Squared Exponental covariance function
likfunc = {@likGauss};              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);

[~, ~ , mu, s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, x);

m_sum = feval(meanfunc{:}, hyp.mean, x);
K_sum = feval(covfunc{:}, hyp.cov, x);
[post, ~, ~] = infExact(hyp, meanfunc, covfunc, likfunc, x, y);
m_post = m_sum + K_sum*post.alpha;
tic;
tmp = K_sum.*post.sW;
K_post = K_sum - solve_chol(post.L,tmp)*tmp';
toc;
if max(abs(mu-m_post))<=1e-10
   disp("pass mean");
else
    disp("fail mean");
end

if max(abs(s2-diag(K_post)))<=1e-10
   disp("pass var");
else
    disp("fail var");
end