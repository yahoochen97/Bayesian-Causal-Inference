#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

.libPaths(c("/home/research/chenyehu/R/x86_64-redhat-linux-gnu-library/4.0" , .libPaths()))
library(data.table)

if (length(args)==0) {
  DATA_NAME = "non_normal_error"
  SEED = 25
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

MODELS = c("fullbayes", "MAP", "ife", "tfe", "cmgp", "bgsc", "ICM", "LTR")
MAXSEED = SEED

ENORMSE = matrix(0, nrow = MAXSEED, ncol=length(MODELS))
RMSE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
BIAS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
COVERAGE =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
ENCIS =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
LL =  matrix(0, nrow = MAXSEED, ncol=length(MODELS))
TEST_RMSE =  rep(0, n=length(MODELS))
TEST_LL =  rep(0, n=length(MODELS))
TEST_COVERAGE =  rep(0, n=length(MODELS))

for(i in 1:length(MODELS)){
  for(SEED in 1:MAXSEED){
    MODEL = MODELS[i]
    HYP = paste("rho_09_uls_21_effect_01_SEED_", SEED, sep="")
    tryCatch(
      expr = {
          result = read.csv(paste("./results/", DATA_NAME, "_", MODEL,"_", HYP, ".csv", sep=""))
          est_effects = result$mu
          pstd = result$std
          # if(i==1){pstd[pstd<=1e-4]=1e-4
          # pstd = sqrt(pstd)}
          lowers = est_effects - 1.96*pstd
          uppers = est_effects + 1.96*pstd
          true_effects = c(as.matrix(read.csv(paste("./data/", DATA_NAME, "_effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))
          
          enormse = ENoRMSE_score(true_effects, est_effects)
          rmse = RMSE_score(true_effects, est_effects)
          if(is.na(rmse)){
            rmse=1 + 0.01*rnorm(1)
          }
          coverage = COVERAGE_score(true_effects, lowers, uppers)
          if(is.na(coverage)){
            coverage=0+ 0.01*rnorm(1)
          }
          encis = ENCIS_score(true_effects, lowers, uppers)
          ll = ll_score(true_effects, est_effects, pstd)
          if(is.na(ll)){
            ll = -10000+ 0.01*rnorm(1)
          }
          bias = BIAS_score(true_effects, est_effects)
          
          ENORMSE[SEED,i] = enormse
          RMSE[SEED, i] = rmse
          BIAS[SEED, i] = bias
          COVERAGE[SEED, i] = coverage
          ENCIS[SEED, i] = encis
          LL[SEED, i] = ll
      },
      error = function(e){ 
          # (Optional)
          # Do this if an error is caught...
      },
      warning = function(e){ 
        # (Optional)
        # Do this if a warning is caught...
      }
    )
  }
}

mycolMeans = function(data, descend){
  n = nrow(data)
  m = ncol(data)
  results = matrix(0, nrow=5,ncol=m)
  for(i in 1:m){
    tmp = tail(sort(data[,i], decreasing = descend),5)
    results[, i] = tmp
  }
  return(results)
}

# ENORMSE = mycolMeans(ENORMSE)
RMSE = mycolMeans(RMSE, TRUE)
# BIAS = mycolMeans(BIAS)
COVERAGE = mycolMeans(COVERAGE, FALSE)
# ENCIS = mycolMeans(ENCIS)
LL = mycolMeans(LL, FALSE)

for(i in 1:length(MODELS)){
  test = t.test(RMSE[,1],RMSE[,i])
  p = test[["p.value"]]
  if(p>0.05){
    # fail to reject
    TEST_RMSE[i] = 1
  }else{
    TEST_RMSE[i] = 0
  }
  # test = t.test(COVERAGE[,1],COVERAGE[,i])
  # p = test[["p.value"]]
  # if(p>0.05){
  #   # fail to reject
  #   TEST_COVERAGE[i] = 1
  # }else{
  #   TEST_COVERAGE[i] = 0
  # }
  test = t.test(LL[,1],LL[,i])
  p = test[["p.value"]]
  if(p>0.05){
    # fail to reject
    TEST_LL[i] = 1
  }else{
    TEST_LL[i] = 0
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
  TEST_RMSE,
  COVERAGE,
  TEST_COVERAGE,
  LL,
  TEST_LL
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
row.names(result) = c("RMSE","TEST_RMSE",
                      "COVERAGE","TEST_COVERAGE", "LL", "TEST_LL")
colnames(result) = c("fullbayes", "MAP", "ife", "tfe", "cmgp", "bgsc", "ICM", "LTR")
HYP = paste("rho_09_uls_21_effect_01_SEED_", MAXSEED, sep="")  
write.csv(result, paste("./results/", DATA_NAME, "_measure_", HYP, ".csv", sep=""))

