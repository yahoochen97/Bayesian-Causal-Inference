library(data.table)
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

setwd("./data/synthetic")

if (length(args)==0) {
  SEED = 1
}
if (length(args)==1) {
  SEED =args[1]
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

ll_score = function(true_effects, est_effects, pstd){
  score = -0.5*(true_effects-est_effects)**2/pstd**2 - 0.5*log(2*pi*pstd**2)
  return(mean(score))
} 

MODELS = c("multigp", "ife", "tfe")

ENORMSE = c()
RMSE = c()
BIAS = c()
COVERAGE = c()
CIC = c()
ENCIS = c()
LL = c()

for(MODEL in MODELS){
  result = read.csv(paste(MODEL, "_", SEED, ".csv", sep=""))
  est_effects = result$mu
  pstd = result$std
  lowers = est_effects - 1.96*pstd
  uppers = est_effects + 1.96*pstd
  true_effects = c(as.matrix(read.csv(paste("effect_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE)))
  
  enormse = ENoRMSE_score(true_effects, est_effects)
  rmse = RMSE_score(true_effects, est_effects)
  bias = BIAS_score(true_effects, est_effects)
  coverage = COVERAGE_score(true_effects, lowers, uppers)
  cic = CIC_score(true_effects,est_effects, lowers, uppers)
  encis = ENCIS_score(true_effects, lowers, uppers)
  ll = ll_score(true_effects, est_effects, pstd)
  
  
  ENORMSE = c(ENORMSE, enormse)
  RMSE = c(RMSE, rmse)
  BIAS = c(BIAS, bias)
  COVERAGE = c(COVERAGE, coverage)
  CIC = c(CIC, cic)
  ENCIS = c(ENCIS, encis)
  LL = c(LL, ll)
}

result = data.frame(
  ENORMSE,
  RMSE,
  BIAS,
  COVERAGE,
  CIC,
  ENCIS,
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

result = dfDigits(result, 4)

row.names(result) = c("ENORMSE", "RMSE",
                      "BIAS", "COVERAGE",
                      "CIC","ENCIS", "LL")
colnames(result) = MODELS
write.csv(result, paste("measures_", SEED, ".csv", sep=""))

