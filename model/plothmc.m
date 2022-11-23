addpath("./results");
addpath("./model");
addpath("/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
% plot posteriors
clear chains;
for i=1:5
   load("results/drift" + int2str(i) + ".mat");
   chains{i} = chain;
end

diagnostics(hmc, chains);
samples = vertcat(chains{:});

c = exp(samples);
c(:, 3) = 2 * normcdf(samples(:, 3)) - 1;

% figure(4);
% clf;
% plotmatrix(c);
% close;

fig = figure(4);
clf;
% h = histfit(c(:,3), 78, 'beta');
% h = histogram(c(:,3));
[f,xi] = ksdensity(c(:,3)); 
plot(xi,f, 'r', 'LineWidth', 2);

xlim([min(xi),1]);
FONTSIZE = 16;
xlabel("group correlation {\rho}", 'FontSize',FONTSIZE);
% ylabel("posterior density", 'FontSize',FONTSIZE);
legend("Posterior density",...
        'Location', 'northwest','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');
set(gca(),'box','off','ytick', []);
ax = gca;
ax.YAxis.Visible = 'off';
set(gca(), 'LooseInset', get(gca(), 'TightInset'));

set(fig, 'PaperPosition', [-1 -1 6 3]); 
set(fig, 'PaperSize', [6 3]); 

filename = "./results/localnewsposterior_rho.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;

% plot posteriors
clear chains;
load("./results/plotsigact.mat");

c = exp(chain);
c(:, 3) = 2 * normcdf(chain(:, 3)) - 1;
c(:, 4) = log(c(:,4));

% figure(4);
% clf;
% plotmatrix(c);
% close;

fig = figure(4);
clf;
% h = histfit(c(:,3), 78, 'beta');
% h = histogram(c(:,3));

[f,xi] = ksdensity(c(:,3)); 
plot(xi,f, 'r', 'LineWidth', 2);

xlim([min(xi),1]);
FONTSIZE = 16;
legend("Posterior density of {\rho}",...
        'Location', 'northwest','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');
set(gca(),'box','off');
set(gca(), 'LooseInset', get(gca(), 'TightInset'));

set(fig, 'PaperPosition', [-1 -1 6 6]); 
set(fig, 'PaperSize', [5 5]); 

filename = "./results/sigactposterior_rho.pdf";
print(fig, filename, '-dpdf','-r300', '-fillpage');
close;
