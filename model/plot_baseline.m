SEED = 1;
uls = 100;
rho = 0.999;
effect = 0.1;
addpath("data/synthetic/")

MODELS = ["fullbayes", "ife", "tfe", "cmgp", "bgsc"];

HYP="_rho_"+strrep(num2str(rho),'.','')+"_uls_"+...
    num2str(uls) + "_effect_"+strrep(num2str(effect),'.','')  + "_SEED_" + SEED;

effects = load("effect"+HYP+".csv");
effects = [zeros(30,1); effects'];

days = (1:50)';
for i=1:numel(MODELS)
   data = readtable(MODELS(i)+HYP+".csv"); 
   pmu = zeros(50, 1);
   pmu(31:end) = data.mu;
   pstd = zeros(50, 1);
   pstd(31:end) = data.std;
   fig = figure(1);
   clf;
   f = [(pmu+1.96*pstd); (flip(pmu-1.96*pstd,1))];
   fill([days; flip(days,1)], f, [7 7 7]/8);
   hold on; plot(days, pmu); 
   plot(days, effects, "--");
   title(MODELS(i));
   
   set(fig, 'PaperPosition', [0 0 10 10]); 
   set(fig, 'PaperSize', [10 10]); 

   filename = "data/synthetic/" + MODELS(i) + HYP + "_SEED_" + SEED + ".pdf";
   print(fig, filename, '-dpdf','-r300');
   close;
end