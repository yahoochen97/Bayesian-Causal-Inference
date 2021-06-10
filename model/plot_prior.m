plot_idx = [1:3, 6:7,10, 12, 14:16];
titles = ["group ls", "group os", "rho", "unit ls", "unit os", "day", "weekday", "b", "drift ls", "drift os", "noise"];
figure(4);
clf;

n = 10000;
nbins = 25;
for i=1:numel(plot_idx)
   subplot(3,4,i);
   p = prior.cov{plot_idx(i)};
   y = zeros(n,1);
   for j=1:n, y(j) = feval(p{:}); end
   if i==3
       y= 2 * normcdf(y) - 1;
   elseif i~=8
       y= exp(y); 
   end
   histogram(y, nbins);
   title(titles(i));
end

i = 1 + numel(plot_idx);
subplot(3,4,i);
p = prior.lik{1};
y = zeros(n,1);
for j=1:n, y(j) = feval(p{:}); end
y = exp(y);
histogram(y, nbins);
title(titles(i));
