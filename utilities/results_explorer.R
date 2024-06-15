library(tidyverse)
library(ggplot2)

ntl_fitted_gpr <- read.csv("~/GitHub/NTL-GP/results/ntl_fitted_gpr.csv")
ntl_train_data <- read.csv("~/GitHub/NTL-GP/results/ntl_train_data.csv")

nrow(ntl_fitted_gpr)==nrow(ntl_train_data)

data <- cbind(ntl_fitted_gpr, ntl_train_data)
head(data)


data %>%
  group_by(Treated, period) %>%
  summarize(group_mean = mean(gpr_mean, na.rm=T)) %>%
  ggplot(aes(x=period, y=group_mean, color=as.factor(Treated))) +
  geom_line()
