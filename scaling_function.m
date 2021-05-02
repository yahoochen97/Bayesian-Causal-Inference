function f = scaling_function(theta, x, i)

  if (nargin < 2)
    f = '2';
    return;
  end

  a = theta(1);
  b = theta(2);

  xx = (x - a) ./ (b - a) - 1;
  in_ind   = ((xx > -1) & (xx < 0));
  after_ind = (xx >= 0);

  f = zeros(size(x));

  if (nargin == 2)
    f(in_ind) = exp(1) * exp(-1 ./ (1 - xx(in_ind).^2));
    f(after_ind) = 1;
    return;
  end

  if (i > 1)
    f(in_ind) = exp(1) * exp(-1 ./ (1 - xx(in_ind).^2)) .* ...
        (2 * (a - b).^2 .* (b - x(in_ind))) ./ ...
        ((a - x(in_ind)).^2 .* (a - 2 * b + x(in_ind)).^2);
  end

end