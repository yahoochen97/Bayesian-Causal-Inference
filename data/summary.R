#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
setwd("./data/synthetic")

MAXSEED = as.integer(args[1])
EFFECT = as.double(args[2])
inputs = as.double(args[-c(1,2)])
ULS = as.integer(inputs[inputs>=1])
RHOS = as.numeric(inputs[inputs<1])

MODELS = c("fullbayes", "multigp", "naivecf", "whitenoise", "grouptrend", 
           "ife", "tfe", "blr", "cmgp", "bgsc")
results = data.frame(matrix(ncol = 5, nrow = 0))
colnames(results) <- c("uls", "rho", "model", "TYPE", "measure")
for (uls in ULS) {
  for (rho in RHOS) {
      HYP = paste("rho_", sub("\\.", "", toString(rho)), '_uls_', ULS, '_effect_', 
                sub("\\.", "", toString(EFFECT)), '_SEED_', MAXSEED, sep="")
      measures = read.csv(paste("measure", "_", HYP, ".csv", sep=""))
      measures$X = as.character(measures$X)
      for(TYPE in measures$X){
        for(MODEL in MODELS){
          results[nrow(results)+1,] = c(uls, rho, MODEL, TYPE, measures[measures$X==TYPE, MODEL])
      }
    }
  }
}

results$uls = as.numeric(results$uls)
results$rho = as.numeric(results$rho)
results$measure = as.numeric(results$measure)


# for(TYPE in measures$X){
#   for (uls in ULS){
#     HYP = paste('_uls_', uls, '_effect_', 
#                 sub("\\.", "", toString(EFFECT)), '_SEED_', MAXSEED, sep="")
#     tmp = results[results$uls==uls & results$TYPE==TYPE, ]
#     ggplot(tmp) +
#       geom_line(aes(x = rho, y = measure, group = model, colour=model)) +
#       ylab(TYPE)
#     ggsave(paste(TYPE, HYP, ".pdf", sep=""))
#   }
# }

for (uls in ULS) {
  ENORMSE = matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  RMSE =  matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  BIAS =  matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  COVERAGE =  matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  ENCIS = matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  LL =  matrix(0, nrow = length(RHOS), ncol=length(MODELS))
  
  for (i in 1:length(RHOS)) {
    rho = RHOS[i]
    HYP = paste("rho_", sub("\\.", "", toString(rho)), '_uls_', ULS, '_effect_', 
                sub("\\.", "", toString(EFFECT)), '_SEED_', MAXSEED, sep="")
    
    measures = read.csv(paste("measure", "_", HYP, ".csv", sep=""))
    measures$X = as.character(measures$X)
    for(j in 1:length(MODELS)){
      MODEL = MODELS[j]
      ENORMSE[i,j] = measures[measures$X=='ENORMSE', MODEL]
      RMSE[i,j] = measures[measures$X=='RMSE', MODEL]
      BIAS[i,j] = measures[measures$X=='BIAS', MODEL]
      COVERAGE[i,j] = measures[measures$X=='COVERAGE', MODEL]
      ENCIS[i,j] = measures[measures$X=='ENCIS', MODEL]
      LL[i,j] = measures[measures$X=='LL', MODEL]
    }
  }
  row.names(ENORMSE) = RHOS
  colnames(ENORMSE) = MODELS
  row.names(RMSE) = RHOS
  colnames(RMSE) = MODELS
  row.names(BIAS) = RHOS
  colnames(BIAS) = MODELS
  row.names(COVERAGE) = RHOS
  colnames(COVERAGE) = MODELS
  row.names(ENCIS) = RHOS
  colnames(ENCIS) = MODELS
  row.names(LL) = RHOS
  colnames(LL) = MODELS
  HYP = paste('_uls_', uls, '_effect_', 
              sub("\\.", "", toString(EFFECT)), '_SEED_', MAXSEED, sep="")
  write.csv(ENORMSE, paste('ENORMSE', HYP, ".csv", sep=""))
  write.csv(RMSE, paste('RMSE', HYP, ".csv", sep=""))
  write.csv(BIAS, paste('BIAS', HYP, ".csv", sep=""))
  write.csv(COVERAGE, paste('COVERAGE', HYP, ".csv", sep=""))
  write.csv(ENCIS, paste('ENCIS', HYP, ".csv", sep=""))
  write.csv(LL, paste('LL', HYP, ".csv", sep=""))
}

