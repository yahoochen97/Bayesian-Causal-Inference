#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

.libPaths(c("/home/research/chenyehu/R/x86_64-redhat-linux-gnu-library/4.0" , .libPaths()))
library(data.table)

setwd("./data/synthetic")

if (length(args)==0) {
  MAXSEED = 1
  ULS = 21
  RHO = 0.7
  EFFECT  = 0.1
}
if (length(args)==4){
  MAXSEED = as.integer(args[1])
  ULS = as.integer(args[2])
  RHO = as.double(args[3])
  EFFECT = as.double(args[4])
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

MODELS = c("fullbayes", "multigp", "naivecf", "whitenoise", "whitenoisegroup","uncorreffecttrend",
           "grouptrend", "ife", "tfe", "blr", "cmgp", "bgsc", "perfectcor", "naiveICM", "unittrend","individual")

# MODELS = c("fullbayes",  "ife", "tfe", "cmgp", "bgsc", "naiveICM", "unittrend","individual")


ENORMSE = matrix(0, nrow = MAXSEED, ncol=length(MODELS))
RMSE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
BIAS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
COVERAGE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
ENCIS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
LL =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))


for(i in 1:length(MODELS)){
  for(SEED in 1:MAXSEED){
    MODEL = MODELS[i]
    HYP = paste("rho_", sub("\\.", "", toString(RHO)), '_uls_', ULS, '_effect_', 
                sub("\\.", "", toString(EFFECT)), '_SEED_', SEED, sep="")

    result = read.csv(paste(MODEL, "_", HYP, ".csv", sep=""))
    est_effects = result$mu
    pstd = result$std
    lowers = est_effects - 1.96*pstd
    uppers = est_effects + 1.96*pstd
    true_effects = c(as.matrix(read.csv(paste("effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))
    
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
    # ENORMSE = c(ENORMSE, enormse)
    # RMSE = c(RMSE, rmse)
    # COVERAGE = c(COVERAGE, coverage)
    # ENCIS = c(ENCIS, encis)
    # LL = c(LL, ll)
  }
}


# perform paired t-test

# print(paste("correlation: ", RHO , sep=""))
# for (i in 2:length(MODELS)) {
#   tmp = t.test(LL[,1], LL[, i], paired = TRUE)
#   p = tmp[["p.value"]]
#   if (p>=0.05){
#     print(paste(MODELS[i], " not significant worse in LL.", sep=""))
#   }
#   else{
#     print(mean(LL[,1])-mean(LL[,i]))
#   }
# }
# print(paste("correlation: ", RHO , sep=""))
# for (i in c(1,2,3,4)) {
#   tmp = t.test(COVERAGE[,5], COVERAGE[, i], paired = TRUE)
#   p = tmp[["p.value"]]
#   if (p>=0.05){
#     print(paste(MODELS[i], " not significant worse in COVERAGE.", sep=""))
#   }
#   # tmp = t.test(LL[,1], LL[, i], paired = TRUE)
#   # p = tmp[["p.value"]]
#   # if (p>=0.05){
#   #   print(paste(MODELS[i], " not significant worse in LL.", sep=""))
#   # }
# }

ENORMSE_ERR = apply(ENORMSE, 2, sd)/sqrt(MAXSEED)
RMSE_ERR =  apply(RMSE, 2, sd)/sqrt(MAXSEED)
BIAS_ERR =  apply(BIAS, 2, sd)/sqrt(MAXSEED)
COVERAGE_ERR =  apply(COVERAGE, 2, sd)/sqrt(MAXSEED)
ENCIS_ERR = apply(ENCIS, 2, sd)/sqrt(MAXSEED)
LL_ERR =  apply(LL, 2, sd)/sqrt(MAXSEED)

ENORMSE = colMeans(ENORMSE)
RMSE = colMeans(RMSE)
BIAS = colMeans(BIAS)
COVERAGE = colMeans(COVERAGE)
ENCIS = colMeans(ENCIS)
LL = colMeans(LL)


result = data.frame(
  ENORMSE,
  RMSE,
  BIAS,
  COVERAGE,
  ENCIS,
  LL,
  ENORMSE_ERR,
  RMSE_ERR,
  BIAS_ERR,
  COVERAGE_ERR,
  ENCIS_ERR,
  LL_ERR
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

row.names(result) = c("ENORMSE", "RMSE", "BIAS",
                      "COVERAGE", "ENCIS", "LL",
                      "ENORMSE_ERR", "RMSE_ERR", "BIAS_ERR",
                      "COVERAGE_ERR", "ENCIS_ERR", "LL_ERR")
colnames(result) = MODELS
write.csv(result, paste("measure_", HYP, ".csv", sep=""))

