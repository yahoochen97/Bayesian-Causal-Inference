library(ggplot2)
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
setwd("./data/synthetic")

MAXSEED = as.integer(args[1])
EFFECT = as.double(args[2])
ELS = as.integer(args[3])
inputs = as.double(args[-c(1,2,3)])
ULS = as.integer(inputs[inputs>=1])
RHOS = as.numeric(inputs[inputs<1])

MODELS = c("multigp", "ife", "tfe")
results = data.frame(matrix(ncol = 5, nrow = 0))
colnames(results) <- c("uls", "rho", "model", "TYPE", "measure")
for (uls in ULS) {
  for (rho in RHOS) {
      HYP = paste("rho_", sub("\\.", "", toString(RHO)),
                '_uls_', ULS, '_els_', ELS, '_effect_', 
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

# results = results %>%
#   group_by(uls, rho, model, TYPE) %>%
#   summarise(measure=mean(measure))

for(TYPE in measures$X){
  for (uls in ULS){
    HYP = paste('_uls_', ULS, '_els_', ELS, '_effect_', 
                sub("\\.", "", toString(EFFECT)), '_SEED_', MAXSEED, sep="")
    tmp = results[results$uls==uls & results$TYPE==TYPE, ]
    ggplot(tmp) +
      geom_line(aes(x = rho, y = measure, group = model, colour=model)) +
      ylab(TYPE)
    ggsave(paste(TYPE, HYP, ".pdf", sep=""))
  }
}
