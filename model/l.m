function [f, g] = ...
  l(unwrapped_theta, ind, theta, inference_method, mean_function, covariance_function, x, y, likfunction)
  
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