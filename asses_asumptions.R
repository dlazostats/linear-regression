### Linear Regression
library(tidyverse)
library(performance)
library(psych)
library(dplyr)
library(tidymodels)

## data
head(mpg)
pairs.panels(mpg %>% select_if(is.numeric))

## Linear model
model_lm<-linear_reg() %>% 
          set_engine("lm") %>% 
          fit(hwy ~ displ + class, data=mpg)

## Asumptions
check_model(model_lm)
