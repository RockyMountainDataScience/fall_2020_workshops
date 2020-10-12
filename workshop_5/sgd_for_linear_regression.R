library(ggplot2)
library(tidyverse)

data("diamonds")

diamonds %>% filter(cut == "Ideal") %>% 
  ggplot(aes(x = carat, y = price)) + 
  geom_point(alpha = 0.15) + theme_bw() + 
  geom_smooth(method = "lm") + 
  xlab("Carat") + ylab("Price ($)") + 
  ggtitle("Diamond (Ideal) Price vs. Carat") + 
  theme(text = element_text(size=20))

diamonds = diamonds %>% filter(cut == "Ideal")
lm(price ~ carat, data = diamonds)
# int = -2256, beta1 = 7756

set.seed(1)
num_samples = c(5,10,15,25,50,100,150,250,500)
b_init = -1000

err = c()
for(n in 1:length(num_samples)){
  df_sample = diamonds[sample(1:nrow(diamonds), size = num_samples[n]),] %>% select(carat, price)
  
  lm_sample = lm(df_sample$price ~ df_sample$carat)
  m = lm_sample$coefficients[2]
  
  b = c() # a place to store intercept terms at each iteration
  step_size = c() # a place to store step size
  SSR = c() # a place to store SSR values with each iteration
  SSR_der = c() # a place to store the SSR derivative values
  alpha = 0.001 # learning rate - set this manually
  num_iter = 150 # how many iterations
  for(i in 1:num_iter){
    if(i == 1){
      SSR[i] = df_sample %>% mutate(slope = m, pred = slope*carat + b_init, err = pred - price, sq_err = err^2) %>% summarize(ssr = sum(sq_err)) %>% .$ssr
      SSR_der[i] = df_sample %>% mutate(slope = m, der = -2*(price - (b_init + slope*carat))) %>% summarize(der = sum(der)) %>% .$der
      step_size[i] = SSR_der[i]*alpha
      b[i] = b_init - step_size[i]
    }else{
      SSR[i] = df_sample %>% mutate(slope = m, pred = slope*carat + b[i-1], err = pred - price, sq_err = err^2) %>% summarize(ssr = sum(sq_err)) %>% .$ssr
      SSR_der[i] = df_sample %>% mutate(slope = m, der = -2*(price - (b[i-1] + slope*carat))) %>% summarize(der = sum(der)) %>% .$der
      step_size[i] = SSR_der[i]*alpha
      b[i] = b[i-1] - step_size[i]
    }
    # new_intercept = old_intercept - step_size
    #print(paste("Iteration = ", i, "| Intercept = ", round(b[i],2), " | SSR = ", round(SSR[i],2), " | Step size = ", round(step_size[i],2)))
    if(abs(step_size[i]) < 1){break;}
  }
  print(paste0("Estimated intercept = ", round(last(b),2), " | Actual intercept (of subset) = ", round(lm_sample$coefficients[1],2)))
  print(paste0("Intercept (of subset) = ", round(lm_sample$coefficients[1],2), " | True Intercept = ", round(lm(price ~ carat, data = diamonds)$coefficients[1],2)))
  
  err[n] = lm_sample$coefficients[1] - lm(price ~ carat, data = diamonds)$coefficients[1]
}

plot(num_samples, abs(err))

# int = -4433, beta1 = 11132
# Want: y = m*x + b; 
# x = carat, y = price, m = slope
# price = m*carat + intercept
# Assume m = 7756


# get predicted price values assumed b_1 and m (m is fixed)
#df_iter1 = df_sample %>% mutate(pred_price = m*carat + b, err = price - pred_price)
#df_iter1 %>% ggplot(aes(x = carat, y = price)) + geom_point() + theme_bw()

# The error we want is the "sum of squared residuals"

# SSR = (13757 - pred1)^2 + (457 - pred2)^2 + (2321 - pred3)^2
# (1) pred1 = intercept + m*1.51
# (2) pred2 = intercept + m*0.3
# (3) pred3 = intercept + m*0.87
# We have an equation for SSR in terms of all of the individual observation


# Want: change in SSR w.r.t. intercept value
# d(SSR)/d(inter) = d(1)/d(inter) + d(2)/d(inter) + d(3)/d(inter)
# d(1)/d(inter) = (13757 - (intercept + m*1.51))^2 ' =  -2*(13757 - (intercept + m*1.51))
# d(2)/d(inter) = (457 - (intercept + m*0.3))^2 ' = -2*(457 - (intercept + m*0.3))
# d(3)/d(inter) = (2321 - (intercept + m*0.87))^2 ' = -2*(2321 - (intercept + m*0.87))
# (the ' marks above are meant to indicate derivatives)

num_samples = nrow(diamonds)
df_sample = diamonds[sample(1:nrow(diamonds), size = num_samples),] %>% select(carat, price)
# Assuming we know slope term
lm_sample = lm(df_sample$price ~ df_sample$carat)
m = lm_sample$coefficients[2]

# initial guess for intercept term
df_sample %>% ggplot(aes(x = carat, y = price)) + 
  geom_point(size = 5) + theme_bw() + 
  geom_smooth(method = "lm", se = F) + 
  xlab("Carat") + ylab("Price ($)") + 
  ggtitle("Diamond Price vs. Carat") + 
  theme(text = element_text(size=20))

b_init = -1000

b = c() # a place to store intercept terms at each iteration
step_size = c() # a place to store step size
SSR = c() # a place to store SSR values with each iteration
SSR_der = c() # a place to store the SSR derivative values
alpha = 0.00001 # learning rate - set this manually
num_iter = 150 # how many iterations
for(i in 1:num_iter){
  if(i == 1){
    SSR[i] = df_sample %>% mutate(slope = m, pred = slope*carat + b_init, err = pred - price, sq_err = err^2) %>% summarize(ssr = sum(sq_err)) %>% .$ssr
    SSR_der[i] = df_sample %>% mutate(slope = m, der = -2*(price - (b_init + slope*carat))) %>% summarize(der = sum(der)) %>% .$der
    step_size[i] = SSR_der[i]*alpha
    b[i] = b_init - step_size[i]
  }else{
    SSR[i] = df_sample %>% mutate(slope = m, pred = slope*carat + b[i-1], err = pred - price, sq_err = err^2) %>% summarize(ssr = sum(sq_err)) %>% .$ssr
    SSR_der[i] = df_sample %>% mutate(slope = m, der = -2*(price - (b[i-1] + slope*carat))) %>% summarize(der = sum(der)) %>% .$der
    step_size[i] = SSR_der[i]*alpha
    b[i] = b[i-1] - step_size[i]
  }
  # new_intercept = old_intercept - step_size
  print(paste("Iteration = ", i, "| Intercept = ", round(b[i],2), " | SSR = ", round(SSR[i],2), " | Step size = ", round(step_size[i],2)))
  #stopifnot(abs(step_size[i]) > 1) # if we want to put a stop given step size
}
print(paste0("Estimated intercept = ", round(last(b),2), " | Actual intercept (of subset) = ", round(lm_sample$coefficients[1],2)))
print(paste0("Intercept (of subset) = ", round(lm_sample$coefficients[1],2), " | True Intercept = ", round(lm(price ~ carat, data = diamonds)$coefficients[1],2)))

lm_sample$coefficients[1] - lm(price ~ carat, data = diamonds)$coefficients[1]

df_sample %>% 
  mutate(pred = last(b) + carat*m) %>% 
  ggplot(aes(x = carat, y = price)) + 
  geom_point(alpha = 0.5) + theme_bw() + 
  geom_line(aes(x = carat, y = pred), col = 'red') + 
  geom_smooth(method = "lm", se = F) + 
  xlab("Carat") + ylab("Price ($)")

plot(SSR)
plot(SSR_der)
plot(b)
plot(step_size)
plot(b , SSR)

data.frame(b, SSR) %>% 
  ggplot(aes(x = b, y = SSR)) + 
  geom_point() + theme_bw() + 
  xlab("Intercept (b)") + 
  ggtitle("SSR vs. Intercept", subtitle = "Each point is an iteration") + 
  theme(text = element_text(size=20))


