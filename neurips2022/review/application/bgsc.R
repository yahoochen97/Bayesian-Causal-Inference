library(bpCausal)
#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

NUM_INTER = 10

data = read.csv("localnewsdata.csv")

N_tr = length(unique(data[data$group==2, 'id']))
N_co = length(unique(data$id)) - N_tr
T_max = max(data$day)
T0 = 89

effects = read.csv("./localnewsalleffects.csv")$effect

obs_days = read.csv("./localnewsdata.csv")$day

# interactive fixed effect
fit <- bpCausal(data = data, index=c("id","day"), Yname = c('y'),
                Dname = c('D'), Xname = c('weekday'), Zname = NULL, Aname = NULL, 
                re = "two-way", ar1 = TRUE, r=NUM_INTER, 
                xlasso = 1, zlasso = 0, alasso = 0, flasso = 1)
coef_fit = effSummary(fit, usr.id = NULL, cumu = FALSE, rela.period = TRUE)$est.eff
estimated_D = coef_fit[,'estimated_ATT']
lower = coef_fit[,'estimated_ATT_ci_l']
upper = coef_fit[,'estimated_ATT_ci_u']

result = data.frame(estimated_D, (upper-lower)/2/1.96)
names(result) = c("mu", "std")
write.csv(result, paste("../results/localnews_bgsc.csv", sep=""),row.names = FALSE)
