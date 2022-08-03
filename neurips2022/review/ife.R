library(gsynth)
library(estimatr)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  DATA_NAME = "fewer_unit"
  HYP = "rho_09_uls_21_effect_01_SEED_1"
  NUM_INTER = 10
}
if (length(args)==3){
  DATA_NAME = args[1]
  HYP = args[2]
  NUM_INTER = as.integer(args[3])
}

data = read.csv(paste("./data/", DATA_NAME, "_data_", HYP,".csv", sep = ""), row.names = NULL)

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = T_max - sum(data$D)/N_tr

effects = c(as.matrix(read.csv(paste("./data/", DATA_NAME, "_effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))

if(NUM_INTER){
  # interactive fixed effect
  fit <- gsynth(Y=c('y'),D=c('D'), X=c('x1','x2'), data = data, index=c("id","day"),
               CV=TRUE, r = c(1,NUM_INTER), EM=TRUE, force = "two-way", seed=1, se=TRUE, cores=1)
  estimated_D = as.vector(fit$est.att[(T0+1):T_max,'ATT'])
  lower = as.vector(fit$est.att[(T0+1):T_max,'CI.lower'])
  upper = as.vector(fit$est.att[(T0+1):T_max,'CI.upper'])
}else{
  # two way fixed effect
  data$id = as.factor(data$id)
  data$day = as.factor(data$day)
  fit = lm_robust(y ~ 1 + x1 + x2 + id + day + D:day, data = data, clusters = id)
  
  # fit = lm(y ~ 1 + x1 + x2 + id + day + D:day, data = data)
  estimated_D = rep(0, T_max)
  for(i in (T0+1):T_max){
    estimated_D[i] = fit$coefficients[[paste("day", i,":D", sep = "")]]
  }
  estimated_D = estimated_D[(T0+1):T_max]
  out = summary(fit)
  # Std. Errors for treatment effect are the same
  estimated_sd = sapply((T0+1):T_max, function(t){out$coefficients[paste("day", t,":D", sep = ""), 2]})
  lower = estimated_D - 1.96*estimated_sd
  upper = estimated_D + 1.96*estimated_sd
}

result = data.frame(estimated_D, (upper-lower)/2/1.96)
names(result) = c("mu", "std")
if(NUM_INTER==0){
  write.csv(result, paste("./results/", DATA_NAME, "_tfe_", HYP, ".csv", sep=""),row.names = FALSE)
}else{
  write.csv(result, paste("./results/", DATA_NAME, "_ife_", HYP, ".csv", sep=""),row.names = FALSE)
}

