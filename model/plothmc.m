% plot posteriors

for i=1:5
   load("results/drifthmc" + int2str(i) + ".mat");
   chains{i} = chain;
end

diagnostics(hmc, chains);
samples = vertcat(chains{:});

c = exp(samples);
c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;
% c(:, 8) = log(c(:,8));
% c(:, 9) = log(c(:,9));

c(:, 8) = log(c(:,8));

figure(3);
clf;
plotmatrix(c);

