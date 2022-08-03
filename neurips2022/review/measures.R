#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

.libPaths(c("/home/research/chenyehu/R/x86_64-redhat-linux-gnu-library/4.0" , .libPaths()))
library(data.table)

if (length(args)==0) {
  DATA_NAME = "non_normal_error"
  SEED = 1
}
if (length(args)==2){
  DATA_NAME = args[1]
  SEED = as.integer(args[2])
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
  score = mean(abs(est_effects[mask]-true_effects[mask]))
  return(score)
}


COVERAGE_score = function(true_effects, lowers, uppers){
  mask = (true_effects!=0)
  score = mean( (true_effects[mask]>=lowers[mask]) & (true_effects[mask]<=uppers[mask]) )
  return(score)
}

ENCIS_score = function(true_effects, lowers, uppers){
  mask = (true_effects!=0)
  score = mean( abs(uppers[mask]-lowers[mask]) / abs(true_effects[mask]) )
  return(score)
}

ll_score = function(true_effects, est_effects, pstd){
  score = -0.5*(true_effects-est_effects)**2/pstd**2 - 0.5*log(2*pi*pstd**2)
  return(mean(score))
} 

MODELS = c("non_normal_error",
           "non_smooth",
           "fewer_unit",
           "independent_gp")

MODELS = c("fullbayes", "ife", "tfe", "cmgp", "bgsc", "icm", "ltr")
MAXSEED = SEED

ENORMSE = matrix(0, nrow = MAXSEED, ncol=length(MODELS))
RMSE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
BIAS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
COVERAGE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
ENCIS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
LL =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))

for(i in 1:length(MODELS)){
  for(SEED in 1:MAXSEED){
    MODEL = MODELS[i]
    HYP = paste("rho_09_uls_21_effect_01_SEED_", SEED, sep="")
    result = read.csv(paste("./results/", DATA_NAME, "_", MODEL,"_", HYP, ".csv", sep=""))
    est_effects = result$mu
    pstd = result$std
    lowers = est_effects - 1.96*pstd
    uppers = est_effects + 1.96*pstd
    true_effects = c(as.matrix(read.csv(paste("./data/", DATA_NAME, "_effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))
    
    enormse = ENoRMSE_score(true_effects, est_effects)
    rmse = RMSE_score(true_effects, est_effects)
    coverage = COVERAGE_score(true_effects, lowers, uppers)
    encis = ENCIS_score(true_effects, lowers, uppers)
    ll = ll_score(true_effects, est_effects, pstd)
    bias = BIAS_score(true_effects, est_effects)
    
    ENORMSE[SEED,i] = enormse
    RMSE[SEED, i] = rmse
    BIAS[SEED, i] = bias
    COVERAGE[SEED, i] = coverage
    ENCIS[SEED, i] = encis
    LL[SEED, i] = ll
  }
}

ENORMSE = colMeans(ENORMSE)
RMSE = colMeans(RMSE)
BIAS = colMeans(BIAS)
COVERAGE = colMeans(COVERAGE)
ENCIS = colMeans(ENCIS)
LL = colMeans(LL)

result = data.frame(
  RMSE,
  COVERAGE,
  LL
)

result = transpose(result)
dfDigits <- function(x, digits = 2) {
  ## x is a data.frame
  for (col in colnames(x)[sapply(x, class) == 'numeric']){
    x[,col] <- round(x[,col], digits = digits)
  }
  return(x)
}

result = dfDigits(result, 5)
row.names(result) = c("RMSE", 
                      "COVERAGE","LL")
colnames(result) = MODELS
HYP = paste("rho_09_uls_21_effect_01_SEED_", MAXSEED, sep="")  
write.csv(result, paste("./results/", DATA_NAME, "_measure_", HYP, ".csv", sep=""))

