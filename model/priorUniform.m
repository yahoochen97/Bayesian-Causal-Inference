function [lp,dlp] = priorUniform(a,b,x)

% Univariate Uniform hyperparameter prior distribution.
% Compute log-likelihood and its derivative or draw a random sample.
% The prior distribution is parameterized as:
%
%   p(x) = 1/|b-a|, where
%
% a and b are the left/right end point of uniform distribution and
% x(1xN) contains query hyperparameters for prior evaluation.
%
% For more help on design of priors, try "help priorDistributions".
%
% Copyright (c) by Yehu Chen, 2021-06-04.
%
% See also PRIORDISTRIBUTIONS.M.

if nargin<2, error('a and b parameters need to be provided'), end
if ~(isscalar(a)&&isscalar(b))
  error('a and b parameters need to be scalars')
end
if ~(a<b)
  error('a needs to be smaller than b')
end
if nargin<3, lp = log(unifrnd(a,b)); return, end % return a sample
if or(x<a, x>b)
    lp  = -inf;
    dlp = 0;
else
    lp  = log(1/abs(b-a));
    dlp = 0;
end