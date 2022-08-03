library(gsynth)
library(estimatr)

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

NUM_INTER = 10

data = read.csv("localnewsdata.csv")
effects = read.csv("./localnewsalleffects.csv")$effect
obs_days = sort(unique(read.csv("./localnewsdata.csv")$day))

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = 89

if(NUM_INTER){
  # interactive fixed effect
  fit <- gsynth(Y=c('y'),D=c('D'), data = data, index=c("id","day"),
               CV=TRUE, r = c(1,NUM_INTER), EM=TRUE, force = "two-way", seed=1, se=TRUE, cores=1)
  estimated_D = rep(0, T_max)
  lower = rep(0, T_max)
  upper = rep(0, T_max)
  estimated_D[obs_days] = as.vector(fit$est.att[,'ATT'])
  lower[obs_days] = as.vector(fit$est.att[,'CI.lower'])
  upper[obs_days] = as.vector(fit$est.att[,'CI.upper'])
}else{
  # two way fixed effect
  data$id = as.factor(data$id)
  data$day = as.factor(data$day)
  fit = lm_robust(y ~ 1 + weekday + id + day + D:day, data = data, clusters = id)
  estimated_D = rep(0, T_max)
  for(i in obs_days){
    estimated_D[i] = fit$coefficients[[paste("day", i,":D", sep = "")]]
  }
  out = summary(fit)
  # Std. Errors for treatment effect are the same
  estimated_sd = sapply(obs_days, function(t){out$coefficients[paste("day", t,":D", sep = ""), 2]})
  estimated_sd[is.na(estimated_sd)] = 0
  lower = rep(0, T_max)
  upper = rep(0, T_max)
  lower[obs_days] = estimated_D[obs_days] - 1.96*estimated_sd
  upper[obs_days] = estimated_D[obs_days] + 1.96*estimated_sd
}

result = data.frame(estimated_D, (upper-lower)/2/1.96)
names(result) = c("mu", "std")
if(NUM_INTER==0){
  write.csv(result, paste("../results/localnews_tfe.csv", sep=""),row.names = FALSE)
}else{
  write.csv(result, paste("../results/localnews_ife.csv", sep=""),row.names = FALSE)
}

