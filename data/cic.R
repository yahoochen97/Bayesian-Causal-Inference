library(qte)
library(synthdid)
library(estimatr)
setwd("./data/synthetic")

#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

if (length(args)==0) {
  SEED = 1
  NUM_INTER = 0
  ULS = 100
  RHO = 0.999
  EFFECT  = 0.1
}
if (length(args)==5){
  SEED = args[1]
  NUM_INTER = as.integer(args[2])
  ULS = as.integer(args[3])
  RHO = as.double(args[4])
  EFFECT = as.double(args[5])
}

# reading data
# treat = as.matrix(read.csv(paste("treat_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE))
# control = as.matrix(read.csv(paste("control_", SEED, ".csv", sep = ""), row.names = NULL, header=FALSE)
HYP = paste("rho_", sub("\\.", "", toString(RHO)), '_uls_', ULS, '_effect_', 
            sub("\\.", "", toString(EFFECT)), '_SEED_', SEED, sep="")

data = read.csv(paste("data_", HYP,".csv", sep = ""), row.names = NULL)

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = T_max - sum(data$D)/N_tr

effects = c(as.matrix(read.csv(paste("effect_", HYP, ".csv", sep = ""), row.names = NULL, header=FALSE)))

# synthetic did
setup = panel.matrices(data, unit=5, time=3, outcome=6, treatment=7)
X = array(rep(0, 50*30*2), dim=c(30, 50, 2))
for (i in 1:30) {
  for (j in 1:50) {
   X[i,j,1] = data[data$day==j & data$id==i, "x1"]
   X[i,j,2] = data[data$day==j & data$id==i, "x2"]
  }
}
tau.hat = synthdid_estimate(setup$Y, setup$N0, setup$T0, X=X)
se = sqrt(vcov(tau.hat, method='placebo'))
sprintf('point estimate: %1.2f', tau.hat)
sprintf('95%% CI (%1.2f, %1.2f)', tau.hat - 1.96 * se, tau.hat + 1.96 * se)
plot(tau.hat)

# CIC
# fit <- CiC(y~D, xformla = ~x1+x2,
#           t=50, tmin1=31, tname = "day", idname="id", data=data, 
#           se=FALSE, cores=1)
estimated_D = as.vector(fit$est.att[(T0+1):T_max,'ATT'])
lower = as.vector(fit$est.att[(T0+1):T_max,'CI.lower'])
upper = as.vector(fit$est.att[(T0+1):T_max,'CI.upper'])

result = data.frame(estimated_D, (upper-lower)/2/1.96)
names(result) = c("mu", "std")
if(NUM_INTER==0){
  write.csv(result, paste("tfe_", HYP, ".csv", sep=""),row.names = FALSE)
}else{
  write.csv(result, paste("ife_", HYP, ".csv", sep=""),row.names = FALSE)
}

