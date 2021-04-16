function K = covDiscrete2(hyp, x, z, i)

if nargin<2, K = '1'; return; end                  % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode
if xeqz, z = x; end                                % make sure we have a valid z

rho = 2 * normcdf(hyp) - 1;

A = [1, rho; rho, 1];

if (nargin < 4)
  if dg
    K = A(sub2ind([2, 2], x, x));
  else
    K = ones(numel(x), numel(z));
    K(x ~= z') = rho;
  end
else
  drho = 2 * normpdf(hyp);
  dK = [0, drho; drho, 0];
  if dg
    K = dK(sub2ind([2, 2], x, x));
  else
    K = zeros(numel(x), numel(z));
    K(x ~= z') = drho;
  end
end

end
