library(tidyverse)
library(ggthemes)
library(ggplot2)
library(reshape2)
library(dplyr)
setwd("/Users/yahoo/Documents/GitHub/Bayesian-Causal-Inference/data")
## show trends for a selection of stations

segments_dd = read_csv("segments_dd.csv")

station_ids = unique(segments_dd$station_id)

segments_dd$time = paste(segments_dd$date, segments_dd$timeslot, sep=" ")

## some plots
# for (i in 1:length(station_ids)) {
#   station_id = station_ids[i]
#   acquired = segments_dd$sinclair2017[segments_dd$station_id==station_id][1]
#   station_natloc = segments_dd[segments_dd$station_id==station_id, c("date",
#                                                                      "national_politics", "local_politics")]
#   station_natloc = aggregate(.~date, data=station_natloc, mean)
#   station_natloc$date = as.Date(station_natloc$date, format="%m/%d/%Y")
#   station_natloc = station_natloc[order(station_natloc$date),]
#   # station_natloc$national_politics = log(station_natloc$national_politics)
#   # station_natloc$time = paste(station_natloc$date, station_natloc$timeslot, sep=" ")
#   # station_natloc <- reshape2::melt(station_natloc ,  id.vars = 'date', variable.name = 'type')
#   # ggplot(station_natloc, aes(date, 1:length(date))) +
#   #   geom_line(aes(colour = type)) +
#   #   scale_x_date(limits = as.Date(date))
#   ggplot(station_natloc, aes(x=date), group=1) +
#     geom_line(aes(y=national_politics, colour = "red")) +
#     # geom_line(aes(y=local_politics, colour = "blue")) +
#     scale_color_discrete(name = "TYPE", labels = c("national")) +
#     # scale_x_date(limits = date) +
#     ggtitle(paste(station_id, " acquired ", acquired , sep ="")) +
#     theme(plot.title = element_text(hjust = 0.5))
#   ggsave(paste(station_id, ".pdf" , sep =""), height = 20, width = 80, units="cm")
# }

# select subsets
xvars = c("station_id", "date", "national_politics", "sinclair2017", "post","affiliation","timeslot","weekday","callsign")
data = segments_dd[, xvars]

# average for each station on each day
data = aggregate(national_politics~., data=data, mean)
data = subset(data,select=-c(timeslot))
data$t = as.Date(data$date,format='%m/%d/%Y')
data$t = as.vector(data$t - min(data$t)) + 1
  
too_few_station = c()
for (station_id in sort(station_ids)){
  if(nrow(data[data$station_id==station_id,])<=10){
    too_few_station = c(too_few_station, station_id)
  }
}

data = data[!(data$station_id %in% too_few_station),]

write.csv(data, "localnews.csv")

station_ids = unique(data[data$sinclair2017==1, "station_id"])

treat = matrix(data='nan', nrow=length(station_ids), ncol=max(data$t))
for (i in 1:length(station_ids)) {
  tmp = data[data$sinclair2017==1 & data$station_id==station_ids[i],c("national_politics","t")]
  for(t in tmp$t){
    treat[i, t] = tmp[tmp$t==t, "national_politics"]
  }
}

station_ids = unique(data[data$sinclair2017==0, "station_id"])

control = matrix(data='nan', nrow=length(station_ids), ncol=max(data$t))
for (i in 1:length(station_ids)) {
  tmp = data[data$sinclair2017==1 & data$station_id==station_ids[i],c("national_politics","t")]
  for(t in tmp$t){
    control[i, t] = tmp[tmp$t==t, "national_politics"]
  }
}
