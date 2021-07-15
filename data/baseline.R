library(gsynth)
library(dplyr)
setwd("/Users/yahoo/Documents/GitHub/Bayesian-Causal-Inference/data/synthetic")

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  SEED = 1
  NUM_INTER = 5
}
if (length(args)==1){
  SEED = args[1]
  NUM_INTER = 0
}
if (length(args)==2){
  SEED = args[1]
  NUM_INTER = as.integer(args[2])
}

# Shimoni, Yishai, et al. "Benchmarking framework for performance-evaluation 
# of causal inference analysis." arXiv preprint arXiv:1802.05046 (2018).

ENoRMSE_score = function(true_effects, est_effects){
  mask = (true_effects!=0)
  score = sqrt(mean((1-est_effects[mask]/true_effects[mask])**2))
  return(score)
}

RMSE_score = function(true_effects, est_effects){
  mask = (true_effects!=0)
  score = sqrt(mean((est_effects[mask]-true_effects[mask])**2))
  return(score)
}

BIAS_score = function(true_effects, est_effects){
  mask = (true_effects!=0)
  score = mean(est_effects[mask]-true_effects[mask])
  return(score)
}

COVERAGE_score = function(true_effects, lowers, uppers){
  mask = (true_effects!=0)
  score = mean( (true_effects[mask]>=lowers[mask]) & (true_effects[mask]<=uppers[mask]) )
  return(score)
}

CIC_score = function(true_effects,est_effects, lowers, uppers){
  mask = (true_effects!=0)
  score = mean( abs(true_effects[mask]-est_effects[mask]) / abs(uppers[mask]-lowers[mask]) )
  return(score)
}

ENCIS_score = function(true_effects, lowers, uppers){
  mask = (true_effects!=0)
  score = mean( abs(uppers[mask]-lowers[mask]) / abs(true_effects[mask]) )
  return(score)
}

data = read.csv(paste("gsc", ".csv", sep = ""), row.names = NULL, header=FALSE)
colnames(data) <- c("x1","x2","time","group","unit","y","D","effect")

# treat = as.matrix(read.csv(paste("treat_", SEED, ".csv", sep = ""), header = FALSE))
# control = as.matrix(read.csv(paste("control_", SEED, ".csv", sep = ""), header = FALSE))
# effect = as.matrix(read.csv(paste("effect_", SEED, ".csv", sep = ""), header = FALSE))
# effect = colMeans(effect)
# names(effect) <- NULL
N_tr = length(unique(data[data$group==2, 'unit']))
N_co = length(unique(data$unit)) - N_tr
T_MAX = max(data$time)
T0 = T_MAX - sum(data$D)/N_tr
effect = data[data$D==1,c('time','effect')] %>% 
  dplyr::group_by(time) %>%
  summarise(tmp=mean(effect)) %>%
  arrange(time) %>%
  pull(tmp)

# two way fixed effect
if(NUM_INTER==0){
  data$unit = as.factor(data$unit)
  data$time = as.factor(data$time)
  fit = lm(y ~ 1 + x1 + x2 + unit + time + D:time, data = data)
  estimated_D = rep(0, T_MAX)
  for(i in (T0+1):T_MAX){
    estimated_D[i] = fit$coefficients[[paste("time", i,":D", sep = "")]]
  }
  estimated_D = estimated_D[(T0+1):T_MAX]
  out = summary(fit)
  # Std. Errors for treatment effect are the same
  estimated_sd = sapply((T0+1):T_MAX, function(t){out$coefficients[paste("time", t,":D", sep = ""), 2]})
  lower = estimated_D - 1.96*estimated_sd
  upper = estimated_D + 1.96*estimated_sd
  # print(summary(fit))
}

if(NUM_INTER){
  # interactive fixed effect
  fit <- gsynth(Y=c('y'),D=c('D'),X=c('x1','x2'), data = data, index=c("unit","time"),
                 r = c(0,NUM_INTER), CV=TRUE, force = "two-way", nboots = 200, seed=1,
                se=TRUE)
  estimated_D = fit$est.att[(T0+1):T_MAX,'ATT']
  lower = fit$est.att[(T0+1):T_MAX,'CI.lower']
  upper = fit$est.att[(T0+1):T_MAX,'CI.upper']
}

result = data.frame(effect, estimated_D, lower, upper)
names(result) = c("effect", "estimated_D", "lower", "upper")
# write.csv(result, paste("fixedeffect_", SEED, "_r", NUM_INTER, ".csv", sep=""))
print(effect)
print(estimated_D)
print(lower)
print(upper)

