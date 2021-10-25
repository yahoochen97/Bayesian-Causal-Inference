function K = covDiscreteICM(N, J, hyp, x, z, i)

% Covariance function for discrete inputs. Given a function defined on the
% integers 1,2,3,..,N, the covariance function is parameterized as:
%
% k(x,z) = K_{xz} = <BETA_x, BETA_z>, where BETA is a matrix of size N x J.
%
% where K is a matrix of size (N x N).
%
% This implementation assumes that the inputs x and z are given as integers
% between 1 and N.
%
% The hyperparameters specify the BETA matrix of size N x J.
%
% hyp = [ (BETA_11)
%         (BETA_21)
%         ...
%         (BETA_N1)
%         (BETA_12)
%         ..
%         (BETA_NJ)]
%
% The hyperparameters hyp can be generated from BETA using:
% hyp = reshape(BETA,[1,N*J]);
%
% The covariance matrix K is obtained from the hyperparameters hyp by:
% BETA = reshape(hyp, [N,J]); K = BETA*BETA';
%
% This parametrization allows unconstrained optimization of K.
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Yehu Chen, 2021-10-25.
%
% See also MEANDISCRETE.M, COVFUNCTIONS.M.

if nargin<=1, error('N and J must be specified.'), end           % check for dimension
if nargin<4, K = num2str(N*J); return; end   % report number of parameters
if nargin<5, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode
if xeqz, z = x; end                                % make sure we have a valid z

if size(hyp, 2) == 1, hyp = hyp'; end
BETA = reshape(hyp, [N,J]);                        % build BETA matrix

if nargin<6
  A = BETA*BETA'; % A is a placeholder for K to avoid a name clash with the return arg
  if dg
    K = A(sub2ind(size(A),x,x));
  else
    K = A(x,z);
  end
else
  col = ceil(i/N); row = i-(col-1)*N;    % indices by tri-root
  dK = zeros(N,N);
  dK(row,:) = dK(row,:) + BETA(:,col)';
  dK(:, row) = dK(:, row) + BETA(:,col);
  if dg
    K = dK(sub2ind(size(dK),x,x));
  else
    K = dK(x,z);
  end
end