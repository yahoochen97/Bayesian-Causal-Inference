function K = covDiscreteIndividual(N, hyp, x, z, i)

% Covariance function for individual GP. Given a function defined on the
% time x integers as t x [1,2,3,..,N], the covariance function is parameterized as:
%
% k(t,x,t',z') = I[x==z] K(t,t';hyp_x).
%
% where K is a covariance on t.
%
% This implementation assumes that the inputs x and z are given as integers
% between 1 and N.
%
% The hyperparameters specify hyp_x.
%
% hyp = [ hyp_1;
%         hyp_2;
%         ...
%         hyp_N]
%
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Copyright (c) by Yehu Chen, 2022-4-26.
%
% See also MEANDISCRETE.M, COVFUNCTIONS.M.

if nargin<1, error('N must be specified.'), end           % check for dimension
if nargin<3, K = num2str(2*N); return; end               % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                   % determine mode
if xeqz, z = x; end                                % make sure we have a valid z

if size(hyp, 1) == 1, hyp = hyp'; end   % make sure hyp is 2N by 1

if nargin<5
    B = (x(:,2)==z(:,2)'); % index matrix
    A = zeros(size(B));
  for k=1:size(x,1)
      for j=1:size(z,1)
          if x(k,2)==z(j,2)
             A(k,j) = covSEiso(hyp([2*x(k,2)-1,2*x(k,2)]), x(k,1), z(j,1)); 
          end
      end
  end
  K = A.*B;
else
  idx = ceil(i/2);
  i_hyp = i - 2*(idx-1);
  B = (x(:,2)==z(:,2)') .* (x(:,2)==idx); % index matrix
  A = covSEiso(hyp([2*idx-1,2*idx]), x(:,1), z(:,1), i_hyp); % gradient
  K = A.*B;
end

end