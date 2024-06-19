library(tidyverse)
library(ggplot2)

ntl_fitted_gpr <- read.csv("~/GitHub/NTL-GP/results/ntl_fitted_gpr.csv")
ntl_train_data <- read.csv("~/GitHub/NTL-GP/results/ntl_train_data.csv")

nrow(ntl_fitted_gpr)==nrow(ntl_train_data)

data <- cbind(ntl_fitted_gpr %>% select(-period), ntl_train_data)
head(data)


data %>%
  group_by(Treated, period) %>%
  summarize(group_mean = mean(gpr_mean, na.rm=T),
            group_lower = mean(gpr_lwr, na.rm=T),
            group_upper = mean(gpr_upr, na.rm=T)) %>%
  ggplot() +
  #geom_ribbon(aes(x=period, ymin=group_lower, ymax=group_upper), fill = "grey70", alpha = 0.3) + 
  geom_line(aes(x=period, y=group_mean, color=as.factor(Treated))) +
  geom_vline(xintercept = 8) +
  theme_minimal()

means <- data %>%
  group_by(Treated, period) %>%
  summarize(group_mean = mean(gpr_mean, na.rm=T),
            group_lower = mean(gpr_lwr, na.rm=T),
            group_upper = mean(gpr_upr, na.rm=T))


ggplot() +
  #geom_ribbon(data=means, aes(x=period, ymin=group_lower, ymax=group_upper), fill = "grey70", alpha = 0.3) + 
  geom_line(data=data, aes(x=period, y=gpr_mean, color=as.factor(Treated), group = obs_id), alpha = 0.1) +
  geom_line(data=means, aes(x=period, y=group_mean, color=as.factor(Treated)), linewidth=2) + 
  geom_vline(xintercept = 8) +
  theme_minimal()

ggplot() +
  geom_line(data=data, aes(x=period, y=t1_mean-t0_mean))
