
### Load Packages --------------------------------------------------------------

library(tidyverse)
library(lubridate)
library(forecast)

### Load Data ------------------------------------------------------------------


deaths_mt <-
  read_csv('Repositories/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')

confirmed_mt <- 
  read_csv('Repositories/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')

### Clean Data -----------------------------------------------------------------

deaths_mt <-
  deaths_mt %>%
  filter(Province_State == "Montana", Admin2 == "Gallatin") %>%
  select(-c(1:11)) %>%
  pivot_longer(cols = everything(), names_to = 'Date', values_to = 'Deaths') %>%
  mutate(Date = parse_date_time(Date, 'mdy')) %>%
  filter(!is.na(Date)) %>%
  mutate(`New Reported Deaths` = `Deaths` - lag(`Deaths`))

confirmed_mt <- 
  confirmed_mt %>%
  filter(Province_State == "Montana", Admin2 == "Gallatin") %>%
  select(-c(1:11)) %>%
  pivot_longer(cols = everything(), names_to = 'Date', values_to = 'Confirmed Cases') %>%
  mutate(Date = parse_date_time(Date, 'mdy')) %>%
  mutate(`New Confirmed Cases` = `Confirmed Cases` - lag(`Confirmed Cases`))

covid_mt <-
  full_join(deaths_mt, confirmed_mt) %>%
  filter(Date > parse_date_time('01/22/2020', 'mdy'))

### COVID-19 Cases by Day Graphic ----------------------------------------------

covid_mt %>%
  ggplot(aes(x = Date, y = `New Confirmed Cases`)) +
  geom_col() +
  labs(title = 'New Confirmed COVID-19 Cases by Day',
       subtitle = 'Data through October 25th, 2020') 

covid_mt %>%
  ggplot(aes(x = Date, y = `New Confirmed Cases`)) +
  geom_point(alpha = 0.35) +
  labs(title = 'New Confirmed COVID-19 Cases by Day',
       subtitle = 'Data through October 25th, 2020')

### Holt-Winters Exponential Smooothing ----------------------------------------

hw_fit <- HoltWinters(covid_mt$`New Confirmed Cases`, gamma = FALSE)

hw_tibble <- tibble(
  Date = covid_mt$Date[3:277],
  `Fitted Values` = hw_fit$fitted[,1],
  `Level Values` = hw_fit$fitted[,2],
  `Trend Values` = hw_fit$fitted[,3]
)

covid_mt %>%
  ggplot(aes(x = Date, y = `New Confirmed Cases`)) +
  geom_col(alpha = 0.5) +
  labs(title = "Holt's Linear Trend Method Applied",
       subtitle = 'Point predictions shown in Red') +
  geom_line(aes(y = `Fitted Values`), data = hw_tibble, color = 'red')

### Decomposing Holt-Winters Output --------------------------------------------

hw_tibble <- tibble(
  Date = covid_mt$Date[3:277],
  `Fitted Values` = hw_fit$fitted[,1],
  `Level Values` = hw_fit$fitted[,2],
  `Trend Values` = hw_fit$fitted[,3]) %>%
  pivot_longer(
    c(`Fitted Values`, `Level Values`, `Trend Values`), 
    names_to = 'Type', 
    values_to = 'Value'
    )

hw_tibble %>%
  ggplot(aes(x = Date, y = Value, color = Type)) +
  facet_grid(rows = vars(Type)) +
  geom_line()

### Simulating AR1 Series ------------------------------------------------------


set.seed(12345)

n_length <- 100
n_series <- 5

simulate_ar1 <- function(n_series, n_length, phi){
  result <- NULL
  for(i in 1:n_series){
    series <- rep(0, n_length)
    for(j in 2:n_length){
      series[j] <- series[j - 1] * phi + rnorm(1)
    }
    result <-
      bind_rows(
        result,
        tibble(
          Series = i,
          Phi = phi,
          Index = 1:n_length,
          Value = series
        )
      )
  }
  
  return(result)
}

high_ar <- simulate_ar1(3, 100, 0.9) %>%
  mutate(Type = 'Phi = 0.9')
low_ar <- simulate_ar1(3, 100, 0.2) %>%
  mutate(Type = 'Phi = 0.2')
negative_ar <- simulate_ar1(3, 100, -0.5) %>%
  mutate(Type = 'Phi = -0.5')

ar_tibble <-
  bind_rows(
    high_ar,
    bind_rows(
      low_ar,
      negative_ar
    )
  ) %>%
  mutate(Series = as_factor(Series))

ar_tibble %>%
  ggplot(aes(x = Index, y = Value, color = Series)) +
  facet_grid(rows = vars(Type)) +
  geom_line() +
  labs(title = 'Simulated AR(1) Series with Varying Phi')

### Simulating MA1 Series ------------------------------------------------------


set.seed(12345)

n_length <- 100
n_series <- 5

simulate_ma1 <- function(n_series, n_length, theta){
  result <- NULL
  for(i in 1:n_series){
    series <- rep(0, n_length)
    errors <- rnorm(100, mean = 0, sd = 1)
    for(j in 2:n_length){
      series[j] <- errors[j] + theta * errors[j - 1]
    }
    result <-
      bind_rows(
        result,
        tibble(
          Series = i,
          Theta = theta,
          Index = 1:n_length,
          Value = series
        )
      )
  }
  
  return(result)
}

high_ma <- simulate_ma1(3, 100, 0.9) %>%
  mutate(Type = 'Theta = 0.9')
low_ma <- simulate_ar1(3, 100, 0.2) %>%
  mutate(Type = 'Theta = 0.2')
negative_ma <- simulate_ar1(3, 100, -0.5) %>%
  mutate(Type = 'Theta = -0.5')

ma_tibble <-
  bind_rows(
    high_ma,
    bind_rows(
      low_ma,
      negative_ma
    )
  ) %>%
  mutate(Series = as_factor(Series))

ma_tibble %>%
  ggplot(aes(x = Index, y = Value, color = Series)) +
  facet_grid(rows = vars(Type)) +
  geom_line()

### Simulating Non-Stationary Series -------------------------------------------


set.seed(54321)

n_length <- 100
n_series <- 5

simulate_ar1 <- function(n_series, n_length, phi){
  result <- NULL
  for(i in 1:n_series){
    series <- rep(0, n_length)
    for(j in 2:n_length){
      series[j] <- series[j - 1] * phi + rnorm(1)
    }
    result <-
      bind_rows(
        result,
        tibble(
          Series = i,
          Phi = phi,
          Index = 1:n_length,
          Value = series
        )
      )
  }
  
  return(result)
}

high_ar <- simulate_ar1(3, 15, 1) %>%
  mutate(Type = 'Phi = 1 (Random walk)')
low_ar <- simulate_ar1(3, 15, 2.5) %>%
  mutate(Type = 'Phi = 2.5')
negative_ar <- simulate_ar1(3, 15, -2.5) %>%
  mutate(Type = 'Phi = -2.5')

ar_tibble <-
  bind_rows(
    high_ar,
    bind_rows(
      low_ar,
      negative_ar
    )
  ) %>%
  mutate(Series = as_factor(Series))

ar_tibble %>%
  ggplot(aes(x = Index, y = Value, color = Series)) +
  facet_grid(rows = vars(Type)) +
  geom_line() +
  labs(title = 'Simulated Non-Stationary Series with Varying Phi')

### Forecasting ARIMA Cases ----------------------------------------------------

arima_fit <- auto.arima(covid_mt$`New Confirmed Cases`)

arima_tibble <- tibble(
  Date = covid_mt$Date[1:277],
  `Fitted Values` = arima_fit$fitted
)

arima_forecast <- forecast(arima_fit)
arima_forecast_tibble <-
  tibble(
    Date = c(
      parse_date_time('10/26/2020', 'mdy'),
      parse_date_time('10/27/2020', 'mdy'),
      parse_date_time('10/28/2020', 'mdy'),
      parse_date_time('10/29/2020', 'mdy'),
      parse_date_time('10/30/2020', 'mdy'),
      parse_date_time('10/31/2020', 'mdy'),
      parse_date_time('11/1/2020', 'mdy'),
      parse_date_time('11/2/2020', 'mdy'),
      parse_date_time('11/3/2020', 'mdy'),
      parse_date_time('11/4/2020', 'mdy')
    ),
    `Point Forecast` = arima_forecast$mean,
    `Upper 95%` = arima_forecast$upper[,2],
    `Lower 95%` = arima_forecast$lower[,2]
  )

covid_mt %>%
  ggplot(aes(x = Date, y = `New Confirmed Cases`)) +
  geom_col(alpha = 0.5) +
  labs(title = "ARIMA Modeling for COVID-19 Case Counts",
       subtitle = 'Fitted values shown in Red; Predictions show in Blue') +
  geom_line(aes(y = `Fitted Values`), data = arima_tibble, color = 'red') +
  geom_line(aes(y = `Point Forecast`), data = arima_forecast_tibble, color = 'blue') +
  geom_line(aes(y = `Lower 95%`), data = arima_forecast_tibble, color = 'blue', linetype = 'dashed') +
  geom_line(aes(y = `Upper 95%`), data = arima_forecast_tibble, color = 'blue', linetype = 'dashed')

### Fitting DLM Model ----------------------------------------------------------

library(dlm)

buildFun <- function(x) {
  dlmModPoly(order = 1, dV = exp(x[1]), dW = exp(x[2]))
}

fit <- dlmMLE(covid_mt$`New Confirmed Cases`, parm = c(0,0), build = buildFun)
dlmFit <- buildFun(fit$par)

filterFit <- dlmFilter(covid_mt$`New Confirmed Cases`, dlmFit)
forecastFit <- dlmForecast(filterFit, nAhead = 10)

## Forecast Fit Tibble
dlm_forecast_tibble <-
  tibble(
    Date = c(
      parse_date_time('10/26/2020', 'mdy'),
      parse_date_time('10/27/2020', 'mdy'),
      parse_date_time('10/28/2020', 'mdy'),
      parse_date_time('10/29/2020', 'mdy'),
      parse_date_time('10/30/2020', 'mdy'),
      parse_date_time('10/31/2020', 'mdy'),
      parse_date_time('11/1/2020', 'mdy'),
      parse_date_time('11/2/2020', 'mdy'),
      parse_date_time('11/3/2020', 'mdy'),
      parse_date_time('11/4/2020', 'mdy')
    ),
    sqrtR = sapply(forecastFit$R, function(x) sqrt(x[1,1])),
    `Point Forecast` = forecastFit$a,
    `Upper 95%` = forecastFit$a + qnorm(0.975, sd = sqrtR),
    `Lower 95%` = forecastFit$a + qnorm(0.025, sd = sqrtR)
  )

v <- unlist(dlmSvd2var(filterFit$U.C, filterFit$D.C))
pl <- dropFirst(filterFit$m) + qnorm(0.05, sd = sqrt(v[-1]))
pu <- dropFirst(filterFit$m) + qnorm(0.95, sd = sqrt(v[-1]))

library(ggplot2)

covid_mt %>%
  ggplot(aes(x = Date, y = `New Confirmed Cases`)) +
  geom_col(alpha = 0.5) +
  labs(title = "Local Level Model for COVID-19 Case Counts",
       subtitle = 'Fitted values shown in Red; Predictions show in Blue') +
  geom_line(aes(y = filterFit$m[2:278]), alpha = 0.75, color = 'red') +
  geom_line(aes(y = pl), alpha = 0.5, color = 'red', linetype = 'dashed') +
  geom_line(aes(y = pu), alpha = 0.5, color = 'red', linetype = 'dashed') +
  geom_line(aes(y = `Point Forecast`), color = 'blue', data = dlm_forecast_tibble) +
  geom_line(aes(y = `Upper 95%`), alpha = 0.5, color = 'blue', alpha = 0.5, linetype = 'dashed', data = dlm_forecast_tibble) +
  geom_line(aes(y = `Lower 95%`), alpha = 0.5, color = 'blue', alpha = 0.5, linetype = 'dashed', data = dlm_forecast_tibble)