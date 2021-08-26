library(gsynth)
library(estimatr)
setwd("./data/synthetic")

.libPaths(c("/home/research/chenyehu/R/x86_64-redhat-linux-gnu-library/3.5" , .libPaths()))

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  SEED = 1
  NUM_INTER = 0
  ULS = 100
  RHO = 0.999
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
# treat = as.matrix(read.csv(paste("treat_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE))
# control = as.matrix(read.csv(paste("control_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE)
HYP = paste("rho_", sub("\\.", "", toString(RHO)), '_uls_', ULS, '_effect_', 
            sub("\\.", "", toString(EFFECT)), '_SEED_', SEED, sep="")

data = read.csv(paste("data_", HYP,".csv", sep = ""), row.names = NULL)

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = T_max - sum(data$D)/N_tr

effects = c(as.matrix(read.csv(paste("effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))

# y = c(c(treat), c(control))
# day = c(rep(1:T_max, each=N_tr), rep(1:T_max, each=N_co))
# id = c(rep(1:N_tr, T_max),rep((1+N_tr):(N_tr+N_co),T_max))
# D = c(rep(0, N_tr*T0), rep(1, N_tr*(T_max-T0)), rep(0, N_co*T_max))
# data = data.frame(y, day, id, D)

# two way fixed effect
# if(NUM_INTER==0){
#   data$id = as.factor(data$id)
#   data$day = as.factor(data$day)
#   fit = lm(y ~ 1 + x1 + x2 + id + day + D:day, data = data)
#   estimated_D = rep(0, T_max)
#   for(i in (T0+1):T_max){
#     estimated_D[i] = fit$coefficients[[paste("day", i,":D", sep = "")]]
#   }
#   estimated_D = estimated_D[(T0+1):T_max]
#   out = summary(fit)
#   # Std. Errors for treatment effect are the same
#   estimated_sd = sapply((T0+1):T_max, function(t){out$coefficients[paste("day", t,":D", sep = ""), 2]})
#   lower = estimated_D - 1.96*estimated_sd
#   upper = estimated_D + 1.96*estimated_sd
# }


if(NUM_INTER){
  # interactive fixed effect
  fit <- gsynth(Y=c('y'),D=c('D'), X=c('x1','x2'), data = data, index=c("id","day"),
                r = c(1,NUM_INTER), CV=TRUE, force = "two-way", seed=1, se=TRUE, cores=1)
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
  write.csv(result, paste("tfe_", HYP, ".csv", sep=""),row.names = FALSE)
}else{
  write.csv(result, paste("ife_", HYP, ".csv", sep=""),row.names = FALSE)
}

