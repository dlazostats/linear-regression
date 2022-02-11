# Linear regression
library(ISLR)
library(jtools)
library(tidyverse)
library(caret)
library(ggplot2)
library(MLmetrics)
library(MASS)
library(gvlma)
library(car)
library(AppliedPredictiveModeling)

# Working directory
script_name <- 'script_linear_reg.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Data
adv<-read.csv("Advertising.csv",sep=",")
dim(adv)

# Train/Test / CV
set.seed(1234)
index<-createDataPartition(adv$Sales,p=0.75,list=F)
train <- adv[index,]
test  <- adv[-index,]
ctrl <- trainControl( method = "repeatedcv",number = 4,repeats = 5)

# Linear regresssion
lm_m<-train(Sales~TV+Radio+Newspaper,
            method="lm",
            data=train,
            trControl=ctrl,
            tuneGrid  = expand.grid(intercept = TRUE))
lm_m
summ(lm_m$finalModel)
par(mfrow=c(2,2))
plot(lm_m$finalModel)
rse<-summary(lm_m$finalModel)$sigma
porc_error<-rse/mean(train$Sales)
porc_error*100

### RG diagnostics
lm_rgm<-lm(Sales~TV+Radio+Newspaper,data=train)

## outliers
outlierTest(lm_rgm)
qqPlot(lm_rgm, main="QQ Plot")
leveragePlots(lm_rgm)

# Influential Observations
avPlots(lm_rgm)
cutoff <- 4/((nrow(adv)-length(lm_rgm$coefficients)-2))
plot(lm_rgm, which=4, cook.levels=cutoff)
influencePlot(lm_rgm, id.method="identify",
              main="Influence Plot",
              sub="Circle size is proportial to Cook's Distance" )

## Non-normality 
qqPlot(lm_rgm, main="QQ Plot")
sresid <- studres(lm_rgm)
hist(sresid, freq=FALSE,
     main="Distribution of Studentized Residuals")
xfit<-seq(min(sresid),max(sresid),length=40)
yfit<-dnorm(xfit)
lines(xfit, yfit)

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(lm_rgm)
spreadLevelPlot(lm_rgm)

# Evaluate Collinearity
vif(lm_rgm)

# Evaluate Nonlinearity
crPlots(lm_rgm)
ceresPlots(lm_rgm)

# Test for Autocorrelated Errors
durbinWatsonTest(lm_rgm)

# Global test of model assumptions
gvmodel <- gvlma(lm_rgm)
summary(gvmodel)


# KNN regress (performace poorly in higher dimension)
set.seed(522)
tGrid <- expand.grid(k = seq(2, 9, by = 1))
knn_m <- train(Sales~TV+Radio+Newspaper,
               method = 'knn',
               data = train,
               trControl=ctrl,
               tuneGrid = tGrid)
plot(knn_m)

## Train results
results <- resamples(list(linear=lm_m, 
                          knn=knn_m))
bwplot(results,metric = "RMSE")
bwplot(results,metric = "MAE")
bwplot(results,metric = "Rsquared")

## Test results
pred_lm<-predict(lm_m,newdata = test)
pred_knn<-predict(knn_m,newdata = test)
testdf<-data.frame(model=c("lm","KNN"),
                   RMSE=c(RMSE(pred_lm,test$Sales),
                          RMSE(pred_knn,test$Sales)))
test %>% 
  mutate(pred_lm=pred_lm,
         pred_knn=pred_knn) %>% 
  select(Sales,pred_knn,pred_lm) %>% 
  gather(var,value,-Sales) %>% 
  ggplot(aes(x=Sales,y=value,color=var))+
  geom_point()+
  geom_smooth(method=lm,se=FALSE)+
  theme_minimal()

### on all data
set.seed(3331)
lm_ma<-train(Sales~TV+Radio+Newspaper,
            method="lm",
            data=adv,
            trControl=ctrl,
            tuneGrid  = expand.grid(intercept = TRUE))
knn_ma <- train(Sales~TV+Radio+Newspaper,
               method = 'knn',
               data = adv,
               trControl=ctrl,
               tuneGrid = tGrid)

pred_lm<-predict(lm_ma,newdata = adv)
pred_knn<-predict(knn_ma,newdata = adv)
alldf<-data.frame(model=c("lm","KNN"),
                   RMSE=c(RMSE(pred_lm,adv$Sales),
                          RMSE(pred_knn,adv$Sales)))
adv %>% 
  mutate(pred_lm=pred_lm,
         pred_knn=pred_knn) %>% 
  select(Sales,pred_knn,pred_lm) %>% 
  gather(var,value,-Sales) %>% 
  ggplot(aes(x=Sales,y=value,color=var))+
              geom_point()+
              geom_smooth(se=FALSE)+
  theme_minimal()

