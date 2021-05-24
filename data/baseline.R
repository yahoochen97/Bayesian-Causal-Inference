library(gsynth)
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

treat = as.matrix(read.csv(paste("treat_", SEED, ".csv", sep = ""), header = FALSE))
control = as.matrix(read.csv(paste("control_", SEED, ".csv", sep = ""), header = FALSE))
effect = as.matrix(read.csv(paste("effect_", SEED, ".csv", sep = ""), header = FALSE))
effect = colMeans(effect)
names(effect) <- NULL
N_tr = dim(treat)[1]
N_co = dim(control)[1]
T_MAX = dim(treat)[2]
T0 = sum(effect==0)

data = read.csv(paste("data_", SEED, ".csv", sep = ""), row.names = NULL)
colnames(data)[4] <- "D"
data$time = data$time + 1
data$unit = as.factor(data$unit)
data$time = as.factor(data$time)

# two way fixed effect
if(NUM_INTER==0){
  fit = lm(y ~ 1 + unit + time + D:time, data = data)
  estimated_D = rep(0, T_MAX)
  for(i in (T0+1):T_MAX){
    estimated_D[i] = fit$coefficients[[paste("time", i,":D", sep = "")]]
  }
  estimated_D = mean(estimated_D[(T0+1):T_MAX])
  out = summary(fit)
  # Std. Errors for treatment effect are the same
  estimated_sd = out$coefficients[paste("time", T_MAX,":D", sep = ""), 2]
  lower = estimated_D - 1.96*estimated_sd
  upper = estimated_D + 1.96*estimated_sd
  # print(summary(fit))
}

if(NUM_INTER){
  # interactive fixed effect
  fit <- interFE(y ~ 1 + D:time, data = data, index=c("unit","time"),
                 r = NUM_INTER, force = "two-way", nboots = 100)
  estimated_D = fit$beta[,1]
  lower = quantile(fit$est.boot[,1],0.025)
  upper = quantile(fit$est.boot[,1],0.975)
}

result = c(mean(effect[(T0+1):T_MAX]), estimated_D, lower, upper)
names(result) = c("effect", "estimated_D", "lower", "upper")
# write.csv(result, paste("fixedeffect_", SEED, "_r", NUM_INTER, ".csv", sep=""))
print(estimated_D)
print(lower)
print(upper)

