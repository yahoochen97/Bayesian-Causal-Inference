addpath("../../code");
addpath("../../code/model");
addpath("../../code/data");
addpath("../../code/gpml-matlab-v3.6-2015-07-07");

startup;

localnews_effects = readtable("./localnewsalleffects.csv");
true_effects = localnews_effects.effect;
days = readtable("./localnewseffects.csv").day;

pstd = readtable("./localnewseffects.csv").pstd;

fprintf("mg-panel first two weeks avg effect: %4f\n", mean(true_effects(89:103)));
fprintf("mg-panel 6 weeks avg effect: %4f\n", mean(true_effects(133:147)));
fprintf("mg-panel 6 weeks avg std: %4f\n", mean((pstd(94:108))));
fprintf("mg-panel last two weeks avg effect: %4f\n", mean(true_effects(178:192)));
fprintf("mg-panel first two weeks avg std: %4f\n", mean((pstd(69:83))));
fprintf("mg-panel last two weeks avg std: %4f\n", mean((pstd(122:136))));
% fprintf("localnews avg effect: %4f\n", mean(true_effects(true_effects~=0)));
% fprintf("localnews avg std: %4f\n", mean(pstd(pstd~=0)));

% plot ICM
localnews_icm = readmatrix("../results/localnews_ICM.csv");
mu = localnews_icm(:,1);
s2 = localnews_icm(:,2).^2;
mu = mu(mu~=0);
s2 = s2(s2~=0);

fprintf("icm first two weeks avg effect: %4f\n", mean(mu(1:11)));
fprintf("icm last two weeks avg effect: %4f\n", mean(mu(34:47)));
fprintf("icm first two weeks avg std: %4f\n", mean(sqrt(s2(1:14))));
fprintf("icm last two weeks avg std: %4f\n", mean(sqrt(s2(34:47))));
fprintf("icm 6 weeks avg effect: %4f\n", mean(mu(29:)));
fprintf("icm 6 weeks avg std: %4f\n", mean(sqrt(s2(29:108))));

% plot ltr
localnews_ltr = readmatrix("../results/localnews_LTR.csv");
mu = localnews_ltr(:,1);
s2 = localnews_ltr(:,2).^2;
mu = mu(mu~=0);
s2 = s2(s2~=0);

fprintf("ltr first two weeks avg effect: %4f\n", mean(mu(1:11)));
fprintf("ltr last two weeks avg effect: %4f\n", mean(mu(63:73)));
fprintf("ltr first two weeks avg std: %4f\n", mean(sqrt(s2(1:11))));
fprintf("ltr last two weeks avg std: %4f\n", mean(sqrt(s2(63:73))));
fprintf("ltr 6 weeks avg effect: %4f\n", mean(mu(31:41)));
fprintf("ltr 6 weeks avg std: %4f\n", mean(sqrt(s2(31:41))));

% plot ife
localnews_ife = readmatrix("../results/localnews_ife.csv");
mu = localnews_ife(:,1);
s2 = localnews_ife(:,2).^2;
mu = mu(mu~=0);
s2 = s2(s2~=0);
fig=figure(1);
clf;
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
% plot(days, true_effects(days), "--");

title('Estimated treatment effects from GSC','FontSize', FONTSIZE);
FONTSIZE = 14;
BIN = 30;
XTICK = BIN*[0:1:abs(210/BIN)];
XTICKLABELS = ["Jun", "Jul", "Aug", "Sept",...
    "Oct", "Nov", "Dec",];

set(gca, 'xtick', XTICK, ...
         'xticklabels', XTICKLABELS,...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, 192]);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'northwest','NumColumns',2,  'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Effect",'FontSize',FONTSIZE);

filename = "./localnews_ife.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
fprintf("gsc first two weeks avg effect: %4f\n", mean(mu(69:83)));
fprintf("gsc last two weeks avg effect: %4f\n", mean(mu(122:136)));
fprintf("gsc first two weeks avg std: %4f\n", mean(sqrt(s2(69:83))));
fprintf("gsc last two weeks avg std: %4f\n", mean(sqrt(s2(122:136))));
fprintf("gsc 6 weeks avg effect: %4f\n", mean(mu(94:108)));
fprintf("gsc 6 weeks avg std: %4f\n", mean(sqrt(s2(94:108))));

% plot tfe
localnews_tfe = readmatrix("../results/localnews_tfe.csv");
mu = localnews_tfe(:,1);
s2 = localnews_tfe(:,2).^2;
mu = mu(mu~=0);
s2 = s2(s2~=0);
mu(isnan(mu)) = 0;
s2(isnan(s2)) = 0;
fig=figure(2);
clf;
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days(4:end); flipdim(days(4:end),1)], f, [7 7 7]/8);
hold on; plot(days(4:end), mu);
% plot(days, true_effects(days), "--");

title('Estimated treatment effects from 2FE','FontSize', FONTSIZE);
set(gca, 'xtick', XTICK, ...
         'xticklabels', XTICKLABELS,...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, 192]);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'northwest','NumColumns',2,  'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Effect",'FontSize',FONTSIZE);

filename = "./localnews_tfe.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
% fprintf("tfe avg effect: %4f\n", mean(mu));
% fprintf("tfe max effect: %4f\n", max(mu));
% fprintf("tfe avg std: %4f\n", mean(sqrt(s2(s2~=0))));
fprintf("tfe first two weeks avg effect: %4f\n", mean(mu(69:83)));
fprintf("tfe last two weeks avg effect: %4f\n", mean(mu(119:133)));
fprintf("tfe first two weeks avg std: %4f\n", mean(sqrt(s2(69:83))));
fprintf("tfe last two weeks avg std: %4f\n", mean(sqrt(s2(119:133))));
fprintf("tfe 6 weeks avg effect: %4f\n", mean(mu(94:108)));
fprintf("tfe 6 weeks avg std: %4f\n", mean(sqrt(s2(94:108))));

% plot bgsc
localnews_bgsc = readmatrix("../results/localnews_bgsc.csv");
mu = localnews_bgsc(:,1);
s2 = localnews_bgsc(:,2).^2;
fig=figure(3);
clf;
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
% plot(days, true_effects(days), "--");

title('Estimated treatment effects from BGSC','FontSize', FONTSIZE);
set(gca, 'xtick', XTICK, ...
         'xticklabels', XTICKLABELS,...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, 192]);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'northwest','NumColumns',2,  'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Effect",'FontSize',FONTSIZE);

filename = "./localnews_bgsc.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
% fprintf("bgsc avg effect: %4f\n", mean(mu));
% fprintf("bgsc max effect: %4f\n", max(mu));
% fprintf("bgsc avg std: %4f\n", mean(sqrt(s2(s2~=0))));

fprintf("bgsc first two weeks avg effect: %4f\n", mean(mu(69:83)));
fprintf("bgsc last two weeks avg effect: %4f\n", mean(mu(122:136)));
fprintf("bgsc first two weeks avg std: %4f\n", mean(sqrt(s2(69:83))));
fprintf("bgsc last two weeks avg std: %4f\n", mean(sqrt(s2(122:136))));
fprintf("bgsc 6 weeks avg effect: %4f\n", mean(mu(94:108)));
fprintf("bgsc 6 weeks avg std: %4f\n", mean(sqrt(s2(94:108))));

% plot cmgp
localnews_cmgp = readmatrix("../results/localnews_cmgp.csv");
mu = localnews_cmgp(:,1);
s2 = localnews_cmgp(:,2).^2;
mu(isnan(mu)) = 0;
s2(isnan(s2)) = 0;
mu = mu(mu~=0);
s2 = s2(s2~=0);
mu = mu(3:end);
s2 = s2(3:end);
mu(1:69) = 0;
s2(1:69) = 0;
fig=figure(4);
clf;
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
fill([days; flipdim(days,1)], f, [7 7 7]/8);
hold on; plot(days, mu);
% plot(days, true_effects(days), "--");

title('Estimated treatment effects from CMGP','FontSize', FONTSIZE);
set(gca, 'xtick', XTICK, ...
         'xticklabels', XTICKLABELS,...
         'XTickLabelRotation',45,...
         'box', 'off', ...
         'tickdir', 'out', ...
    'FontSize',FONTSIZE);
    
xlim([1, 192]);

legend("Effect 95% CI",...
     "Effect mean",...
    'Location', 'southwest','NumColumns',2,  'FontSize',FONTSIZE);
legend('boxoff');
ylabel("Effect",'FontSize',FONTSIZE);

filename = "./localnews_cmgp.pdf";
set(fig, 'PaperPosition', [-1.8 0 22.2 3]); 
set(fig, 'PaperSize', [18.4 3]);
print(fig, filename, '-dpdf','-r300');
close;
% fprintf("cmpg avg effect: %4f\n", mean(mu));
% fprintf("cmgp max effect: %4f\n", max(mu));
% fprintf("cmgp avg std: %4f\n", mean(sqrt(s2(s2~=0))));

fprintf("cmgp first two weeks avg effect: %4f\n", mean(mu(69:83)));
fprintf("cmgp last two weeks avg effect: %4f\n", mean(mu(122:136)));
fprintf("cmgp first two weeks avg std: %4f\n", mean(sqrt(s2(69:83))));
fprintf("cmgp last two weeks avg std: %4f\n", mean(sqrt(s2(122:136))));
fprintf("cmgp 6 weeks avg effect: %4f\n", mean(mu(94:108)));
fprintf("cmgp 6 weeks avg std: %4f\n", mean(sqrt(s2(94:108))));