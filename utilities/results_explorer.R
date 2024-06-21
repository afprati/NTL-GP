library(tidyverse)
library(ggplot2)

data <- read.csv("~/Documents/GitHub/NTL-GP/results/ntl_fitted_gpr.csv")

data %>% 
  group_by(Treated, period) %>% 
  summarize(mean_gpr = mean(gpr_mean),
            mean_raw = mean(true_y),
            mean_ctr = mean(t0_mean)) %>% 
  ggplot() + 
  geom_line(aes(x=period, y=mean_gpr, color=as.factor(Treated))) +
  geom_line(aes(x=period, y=mean_ctr, color=as.factor(Treated)), linetype="dashed") +
  geom_line(aes(x=period, y=mean_raw, color=as.factor(Treated)), linetype="dotted")

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
            group_upper = mean(gpr_upr, na.rm=T),
            t0_mean = mean(t0_mean, na.rm=T))


ggplot() +
  #geom_ribbon(data=means, aes(x=period, ymin=group_lower, ymax=group_upper), fill = "grey70", alpha = 0.3) + 
  geom_line(data=data, aes(x=period, y=gpr_mean, color=as.factor(Treated), group = obs_id), alpha = 0.1) +
  geom_line(data=means, aes(x=period, y=group_mean, color=as.factor(Treated)), linewidth=2) + 
  geom_vline(xintercept = 8) +
  theme_minimal()

ggplot() +
  geom_line(data=means %>% filter(Treated==1), 
            aes(x=period, y=group_mean-t0_mean))


data %>% group_by(Treated, period) %>% summarize(mean_raw = mean(true_y)) %>% ggplot() + geom_line(aes(x=period, y=mean_raw, color=as.factor(Treated)))
data %>% group_by(Treated, period) %>% summarize(mean_gpr = mean(gpr_mean)) %>% ggplot() + geom_line(aes(x=period, y=mean_gpr, color=as.factor(Treated)))

data %>% 
  group_by(Treated, period) %>% 
  summarize(mean_gpr = mean(gpr_mean),
            mean_raw = mean(true_y),
            mean_ctr = mean(t0_mean)) %>% 
  ggplot() + 
  geom_line(aes(x=period, y=mean_gpr, color=as.factor(Treated))) +
  geom_line(aes(x=period, y=mean_ctr, color=as.factor(Treated)), linetype="dashed")


with(data, plot(gpr_mean, true_y))
with(data, cor(gpr_mean, true_y))






temp <- data %>% 
  group_by(Treated, period) %>%
  summarize(mean_raw = mean(true_y),
            mean_gpr = mean(gpr_mean))

