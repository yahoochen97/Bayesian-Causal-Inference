function K = covDiag(meanfunc, hyp, x, z, i)

% Diagonal covariance function, with specified variance.
% The covariance function is specified as:
%
% k(x,z) = var(x) * \delta(x,z)
%
% where var(x) is the noise variance and \delta(p,q) is a Kronecker delta function
% which is 1 iff x=z and zero otherwise.
% var(x) = meanfunc(x, hyp)^2
% The hyperparameter is:
%
% hyp = [ log(meanhyp_1)
%         ...
%         log(meanhyp_n)
%          ]
%

if isempty(meanfunc), K = '0'; return; end  
if ~iscell(meanfunc), meanfunc = {meanfunc}; end

tol = eps;                                                 % threshold on the norm when two vectors are considered to be equal
if nargin<3, K = feval(meanfunc{:}); return; end           % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
dg = strcmp(z,'diag');                                     % determine mode

[n, D] = size(x);
log_noise = feval(meanfunc{:}, hyp, x);
m = exp(2*log_noise);

if dg                                            % vector kxx
  K = m;
  return ;
else
  if isempty(z)  
    K = diag(m);
  else
    nz = size(z, 1);
    flag = (sq_dist(x',z')<sqrt(tol));
    K = repmat(m,1,nz).*flag; % cross covariances Kxz 
  end
end

if nargin<5                                                        % covariances
  return;
else                                                               % derivatives
  d = eval(feval(meanfunc{:}));                                    % number of hyperparameters
  if i>=1 && i<=d
    % derivative w.r.t variance
    dm = feval(meanfunc{:}, hyp, x, i);
    K = 2*repmat(m.*dm,1,nz).*flag;
    K(x~=i, :) = 0;
  else
    error('Unknown hyperparameter')
  end
end

end
