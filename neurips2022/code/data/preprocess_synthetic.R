library(dplyr)
setwd("/Users/yahoo/Documents/GitHub/Bayesian-Causal-Inference/data/synthetic")

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  SEED = 1
  T0 = 40
}
if (length(args)==1){
  SEED = args[1]
  T0 = 40
}
if (length(args)==2){
  SEED = args[1]
  T0 = as.integer(args[2])
}

# reading data
treat = as.matrix(read.csv(paste("treat_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE))
control = as.matrix(read.csv(paste("control_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE))

N_tr = length(treat[,1])
N_co = length(control[,1])
T_max = length(treat[1,])


y = c(c(treat), c(control))
day = c(rep(1:T_max, each=N_tr), rep(1:T_max, each=N_co))
id = c(rep(1:N_tr, T_max),rep((1+N_tr):(N_tr+N_co),T_max))
D = c(rep(1, N_tr*T0), rep(0, N_tr*(T_max-T0)), rep(0, N_co*T_max))

data = data.frame(y, day, id, D)

