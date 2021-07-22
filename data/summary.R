library(ggplot2)
library(dplyr)
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
setwd("./data/synthetic")

MAXSEED = as.integer(args[1])
inputs = as.double(args[-1])
ULS = as.integer(inputs[inputs>=1])
RHOS = as.numeric(inputs[inputs<1])

MODELS = c("multigp", "ife", "tfe")
results = data.frame(matrix(ncol = 5, nrow = 0))
colnames(results) <- c("uls", "rho", "model", "TYPE", "measure")
for (uls in ULS) {
  for (rho in RHOS) {
      HYP = paste("rho_", sub("\\.", "", toString(rho)) , '_uls_', uls, '_SEED_', MAXSEED,  sep="")
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
results$seed = as.numeric(results$seed)
results$measure = as.numeric(results$measure)

# results = results %>%
#   group_by(uls, rho, model, TYPE) %>%
#   summarise(measure=mean(measure))

for(TYPE in measures$X){
  for (uls in ULS){
    tmp = results[results$uls==uls & results$TYPE==TYPE, ]
    ggplot(tmp) +
      geom_line(aes(x = rho, y = measure, group = model, colour=model)) +
      ylab(TYPE)
    ggsave(paste(TYPE, '_uls_', uls, '_SEED_', SEED, ".pdf",  sep=""))
  }
}
