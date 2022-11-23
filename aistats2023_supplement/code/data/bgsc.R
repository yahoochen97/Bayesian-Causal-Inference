library(bpCausal)
setwd("./data/synthetic")

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  SEED = 1
  NUM_INTER = 1
  ULS = 21
  RHO = 0.7
  EFFECT  = 0.1
}
if (length(args)==5){
  SEED = args[1]
  NUM_INTER = as.integer(args[2])
  ULS = as.integer(args[3])
  RHO = as.double(args[4])
  EFFECT = as.double(args[5])
}

# reading data
HYP = paste("rho_", sub("\\.", "", toString(RHO)), '_uls_', ULS, '_effect_', 
            sub("\\.", "", toString(EFFECT)), '_SEED_', SEED, sep="")

data = read.csv(paste("data_", HYP,".csv", sep = ""), row.names = NULL)

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = T_max - sum(data$D)/N_tr

effects = c(as.matrix(read.csv(paste("effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))


# interactive fixed effect
fit <- bpCausal(data = data, index=c("id","day"), Yname = c('y'),
                Dname = c('D'), Xname = c('x1','x2'), Zname = NULL, Aname = NULL, 
                re = "two-way", ar1 = TRUE, r=NUM_INTER, 
                xlasso = 1, zlasso = 0, alasso = 0, flasso = 1)
coef_fit = effSummary(fit, usr.id = NULL, cumu = FALSE, rela.period = TRUE)$est.eff
estimated_D = coef_fit[(T0+1):T_max,'estimated_ATT']
lower = coef_fit[(T0+1):T_max,'estimated_ATT_ci_l']
upper = coef_fit[(T0+1):T_max,'estimated_ATT_ci_u']

result = data.frame(estimated_D, (upper-lower)/2/1.96)
names(result) = c("mu", "std")
write.csv(result, paste("bgsc_", HYP, ".csv", sep=""),row.names = FALSE)
