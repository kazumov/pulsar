# "Pulsar prediction with HTRU2 data set"
# 
# Capstone project
#
# University: HarvardX
# Program: Data Science Professional
# Course: PH125.9x
#
# Copyright: "Ruben R. Kazumov"<kazumov@gmail.com>
# Date: "6/6/2019"
# Version: 1.0
# License: MIT
#

library(tidyverse)
library(readr)
library(ggplot2)
library(knitr)
library(latex2exp)

# Load the data set ----

# If you cloned the project from github, the data file
# `htru2.Rds` is already in your project directory
#
# If you downloaded `report.R` file from the edx server,
# the following code will download the data source file 
download.file(url='https://github.com/kazumov/pulsar/htru2.Rds', 
              destfile ="htru2.Rds")
stars <- readRDS("htru2.Rds")

str(stars) #preview

# Prepare the captions and labels ----
strings <- list(
  captions = c("Mean of the integrated profile", 
                "Standard deviation of the integrated profile", 
                "Excess kurtosis of the integrated profile",  
                "Skewness of the integrated profile", 
                "Mean of the DM-SNR curve", 
                "Standard deviation of the DM-SNR curve", 
                "Excess kurtosis of the DM-SNR curve",
                "Skewness of the DM-SNR curve",
                "Class of star"),
  short = c("Prof. Avg", "Prof. SD", "Prof. Kurt.", "Prof. Skew.", 
            "DM-SNR Avg", "DM-SNR SD", "DM-SNR Kurt.", "DM-SNR Skew.", "Star Class"),
  plotLabels = c(TeX("$Prof_{\\mu}$"), 
                TeX("$Prof_{\\sigma}$"), 
                TeX("$Prof_{k}$"), 
                TeX("$Prof_{S}$"), 
                TeX("$DM_{\\mu}$"), 
                TeX("$DM_{\\sigma}$"), 
                TeX("$DM_{k}$"), 
                TeX("$DM_{S}$")),
  symbols = c("$Prof_{\\mu}$", "$Prof_{\\sigma}$", "$Prof_{k}$", "$Prof_{S}$", "$DM_{\\mu}$", "$DM_{\\sigma}$", "$DM_{k}$", "$DM_{S}$", ""),
  columns = c("profMu", "profSigma", "profK", "profS", "dmMu", "dmSigma", "dmK", "dmS", "starClass")
)

# Overview the data ----

listOfFeatures <- tibble(`#`= 1:dim(stars)[2], 
       `Feature name` = strings$captions, 
       Symbol = strings$symbols, 
       `Column` = strings$columns)

saveRDS(listOfFeatures, file = "listOfFeatures.Rds")


# * The features ----

scaledFeaturesBoxPlot <- data.frame(scale(stars[1:8])) %>%
  mutate(targetClass = stars$starClass) %>%
  gather(key = "Features", value = "Scaled values", 1:8) %>%
  ggplot(aes(x = `Features`, y = `Scaled values`, color = targetClass)) +
  geom_boxplot() +
  scale_x_discrete(labels = strings$short) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(color = "Star class",
       title = "Features",
       subtitle = "Statistical overview of features",
       x = "Features",
       y = "Scaled values of features",
       caption = "Overview of the features")

saveRDS(scaledFeaturesBoxPlot, file = "scaledFeaturesBoxPlot.Rds")

scaledFeaturesBoxPlot


par(mfrow = c(2, 2)) # 2X2
boxplot(profMu~starClass, data = stars, main = strings$captions[1])
boxplot(profSigma~starClass, data = stars, main = strings$captions[2])
boxplot(profK~starClass, data = stars, main = strings$captions[3])
boxplot(profS~starClass, data = stars, main = strings$captions[4])
featuresBoxPlotForIntegratedProfile <- recordPlot()

saveRDS(featuresBoxPlotForIntegratedProfile, file = "featuresBoxPlotForIntegratedProfile.Rds")

par(mfrow = c(1,1)) # clear grid
plot.new() # clear device



# * Proportion of pulsars and not the pulsars ----

# Chart (too ugly to be included in report, but easy to understand)
stars %>%
  select(starClass) %>%
  ggplot(aes(x = starClass)) +
  geom_bar() +
  labs(title = "Proportion of defined star classes in the data set.",
       x = "Star classes", y = "Count")

# Table
starClassesProportion <- summary(stars$starClass)

saveRDS(starClassesProportion, file = "starClassesProportion.Rds")

starClassesProportion # preview

rm(starClassesProportion)

# * Correlations ----
# * * Numeric ----

corNum <- round(cor(stars %>% mutate(starClass = as.numeric(starClass))), 2) 

saveRDS(corNum, file = "corNum.Rds")

corNum # preview

rm(corNum) # cleanup

# * * Graphical ----

pairs(stars)

library(GGally)

symbols <- list(whiteStar = "NP", blackStar = "P")

starsTmp <- stars

levels(starsTmp$starClass) <- c(symbols$whiteStar, symbols$blackStar)

pairs <- starsTmp %>%
  ggpairs(mapping = aes(color = starClass, alpha = 0.1, fill = starClass),
          lower = list(combo = wrap("facethist", binwidth = 0.25)),
          title = "The data set correlation matrix") 

saveRDS(pairs, file = "pairs.Rds")

pairs # preview

rm(starsTmp, symbols, pairs) # cleanup


# Prediction model ----

library(caret)

library(rbenchmark) # for the processing time calculation

n = dim(stars)[1] # number of observations

set.seed(314159)

trainIdx <- sample(x = n, size = n * 0.9) # 90% of observations

train <- stars[trainIdx, ]

test <- stars[-trainIdx, ]

summary(train$starClass) # check the content of the training set

summary(test$starClass) # check the content of the testing set

x <- train %>% select(-starClass) 

y <- train$starClass

xTest <- test %>% select(-starClass)

yTest <- test$starClass

# * Rborist ----

library(Rborist)

fitRborist <- NULL

set.seed(314159)
fitRborist <- Rborist(x = x, y = y)

saveRDS(fitRborist, file = "fitRborist.Rds") # 9.2MB

yHat <- predict(object = fitRborist, newdata = xTest)

confMatrixRborist <- confusionMatrix(yHat$yPred, yTest)

saveRDS(confMatrixRborist, file = "confMatrixRborist.Rds")

confMatrixRborist # preview

#           Reference
# Prediction    0    1
#          0 1612   29
#          1   12  137

rm(confMatrixRborist)

# * Random Forest ----

library(randomForest)

set.seed(314159)

fitRandomForest <- randomForest(x = x, y = y)

saveRDS(fitRandomForest, file = "fitRandomForestDefault.Rds")

yHat <- predict(object = fitRandomForest, newdata = xTest)

confMatrixRandomForestDefault <- confusionMatrix(yHat, yTest)

saveRDS(confMatrixRandomForestDefault, file = "confMatrixRandomForestDefault.Rds")

confMatrixRandomForestDefault # preview

# * * ntree adjusting ----

ntree <- seq(from = 50, to = 500, by = 10)

nTreeVariants <- lapply(ntree, function(ntree){
  fit <- randomForest(x = x, y = y)
  yHat <- predict(object = fit, newdata = xTest)
  cm <- confusionMatrix(yHat, yTest)
  return(
    data.frame(ntree = ntree, accuracy = cm$overall["Accuracy"])
  )
}) %>% bind_rows()

nTreeSelected <- nTreeVariants %>% 
  group_by(ntree) %>% 
  summarise(maxAccuracy = max(accuracy)) %>%
  arrange(maxAccuracy) %>%
  tail(1) %>% .$ntree

nTreeVariantsPlot <- nTreeVariants %>%
  ggplot(aes(x = ntree, y = accuracy, label = accuracy)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = nTreeSelected, color = "blue") +
  geom_hline(yintercept = max(nTreeVariants$accuracy), color = "red", linetype = "dashed")

nTreeVariantsPlot # preview

saveRDS(nTreeVariantsPlot, file = "nTreeVariantsPlot.Rds")

nTreeSelected # preview value

# * * finding the optimal mtry parameter ----

mtry <- 1:8

mtryVariants <- lapply(mtry, function(mtry){
  fit <- randomForest(x = x, y = y)
  yHat <- predict(object = fit, newdata = xTest, ntree = nTreeSelected, mtry = mtry)
  cm <- confusionMatrix(yHat, yTest)
  return(
    data.frame(mtry = mtry, accuracy = cm$overall["Accuracy"])
  )
}) %>% bind_rows()

mtrySelected <- mtryVariants %>% 
  group_by(mtry) %>% 
  summarise(maxAccuracy = max(accuracy)) %>%
  arrange(maxAccuracy) %>%
  tail(1) %>% .$mtry

mtryVariantsPlot <- mtryVariants %>%
  ggplot(aes(x = mtry, y = accuracy, label = accuracy)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = mtrySelected, color = "blue") +
  geom_hline(yintercept = max(mtryVariants$accuracy), color = "red", linetype = "dashed")

mtryVariantsPlot # preview

saveRDS(mtryVariantsPlot, file = "mtryVariantsPlot.Rds")

mtrySelected # preview value

# * * final randomForest approach with optimized parameters ----

set.seed(314159)

fitRandomForestAdjusted <- randomForest(x = x, y = y, ntree = nTreeSelected, mtry = mtrySelected)

plot(fitRandomForestAdjusted)

saveRDS(fitRandomForestAdjusted, file = "fitRandomForestAdjusted.Rds")

yHat <- predict(object = fitRandomForestAdjusted, newdata = xTest)

confMatrixRandomForestAdjusted <- confusionMatrix(yHat, yTest)

saveRDS(confMatrixRandomForestAdjusted, file = "confMatrixRandomForestAdjusted.Rds")

confMatrixRandomForestAdjusted # preview


# * KNN ----

library(caret)

ct <- trainControl(method = "repeatedcv", repeats = 3)

set.seed(314159)

fitKnn <- train(x = x, y = y, method = "knn", trControl = ct, preProcess = c("center", "scale"), tuneLength = 20)

fitKnn

plot(fitKnn) # preview

saveRDS(plot(fitKnn), file = "fitKnnPlot.Rds")

plot(fitKnn, print.thres = 0.5, type = "S") # preview

knnK <- fitKnn$bestTune$k

yHat <- predict(object = fitKnn, newdata = xTest)

confMatrixKnn <- confusionMatrix(yHat, yTest)

saveRDS(confMatrixKnn, file = "confMatrixKnn.Rds")

confMatrixKnn # preview


# Summary for all the methods ----

# * Speed ----

# ATTENTION! The benchmark() re-runs fitting process 10 times for each mithod. 
# Waiting time is about 15 minutes. Sorry!
bm <- benchmark(
  "Rborist" = {
    bmFitRborist <- Rborist(x = x, y = y)
  },
  "Random Forest" = {
    bmFitRFAdj <- randomForest(x = x, y = y, ntree = nTreeSelected, mtry = mtrySelected)
  },
  "KNN" = {
    bmFitKNN <- train(x = x, y = y, method = "knn", trControl = ct, preProcess = c("center", "scale"), tuneLength = 20)
  },
  replications = 10,
  columns = c('test', 'replications', 'elapsed', 'relative', 'user.self', 'sys.self', 'user.child', 'sys.child')
)

bm # preview

saveRDS(bm, "benchmarkTime.Rds")

rm(bmFitRborist, bmFitRFAdj, bmFitKNN)

# * Accuracy

accuracySummary <- tribble(~test, ~`Not a pulsar`, ~`Pulsar`, ~`Erroneous Pulsar`, ~`Missed Pulsar`,
                           "Rborist",       
                           confMatrixRborist$table[1, 1], 
                           confMatrixRborist$table[2, 2], 
                           confMatrixRborist$table[2, 1], 
                           confMatrixRborist$table[1, 2],
                           "Random Forest", 
                           confMatrixRandomForestAdjusted$table[1, 1], 
                           confMatrixRandomForestAdjusted$table[2, 2], 
                           confMatrixRandomForestAdjusted$table[2, 1], 
                           confMatrixRandomForestAdjusted$table[1, 2],
                           "KNN",           
                           confMatrixKnn$table[1, 1], 
                           confMatrixKnn$table[2, 2], 
                           confMatrixKnn$table[2, 1], 
                           confMatrixKnn$table[1, 2])

saveRDS(accuracySummary, file = "accuracySummary.Rds")


# cleanup 
# will be deleted all objects except testing data and `stars` data set
rm(
  list = setdiff(ls(), c("stars", "test", "xTest", "yTest"))
)

# at this point we have loaded only:
# > ls()
# [1] "stars" "test"  "xTest" "yTest"





# Detailed exploration of errorneous observations ----

# we will study recorded Random Forest fitting model 
# file `fitRandomForestAdjusted.Rds` (411.1KB)

fitRandomForestAdjusted <- readRDS(file = "fitRandomForestAdjusted.Rds")

yHat <- predict(object = fitRandomForestAdjusted, newdata = xTest)

errorIdx <- which(! (yHat == yTest))

stars <- test %>% mutate(err = FALSE) # logical, default value

str(stars) # preview

stars[errorIdx, 10] <- TRUE # errorneous observations marked

stars <- stars %>% mutate(predictedPulsar = yHat) # the prediction result "AS IS"

str(stars)

# results and errors types similar to confusion matrix cells:
resultLevels = c("Not a pulsar", 
                 "Pulsar", 
                 "The star was mistaken for a pulsar.", 
                 "Pulsar was missed by the classifier.")


starsAndErrors <- stars %>% 
  mutate(result = case_when(
  err == TRUE & starClass == "not a pulsar" ~ resultLevels[3], # it is actually not a pulsar, but we made a mistake
  err == TRUE & starClass == "pulsar" ~ resultLevels[4], # it is actually pulsar, but we made a mistake
  err == FALSE & starClass == "not a pulsar" ~ resultLevels[1], # it is not a pulsar
  err == FALSE & starClass == "pulsar" ~ resultLevels[2], # it is pulsar
))

starsAndErrors # 1790 lines

saveRDS(starsAndErrors, file = "starsAndErrors.Rds")

starsAndErrors %>% 
  ggplot(aes(x = profMu, y = starClass, color = result)) +
  geom_point(size = 1)


starsAndErrors %>% 
  ggplot(aes(x = dmS, y = starClass, color = result)) +
  geom_point(size = 1)


profSDmSPlot <- starsAndErrors %>% 
  arrange(result) %>%
  ggplot(aes(x = profS, y = dmS, color = result)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("#dddddd", "#aaaaff", "blue", "red")) +
  labs(title = "Errors in prediction",
       subtitle = "Observations on profS over dmS plot.",
       color = "Classification result")

profSDmSPlot # preview

saveRDS(profSDmSPlot, file = "profSDmSPlot.Rds")

profKDmKPlot <- starsAndErrors %>% 
  arrange(result) %>%
  ggplot(aes(x = profK, y = dmK, color = result)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("#dddddd", "#aaaaff", "blue", "red")) +
  labs(title = "Errors in prediction",
       subtitle = "Observations on profK over dmK plot.",
       color = "Classification result")

profKDmKPlot # preview

saveRDS(profKDmKPlot, file = "profKDmKPlot.Rds")

profMuDmMuPlot <- starsAndErrors %>% 
  arrange(result) %>%
  ggplot(aes(x = profMu, y = dmMu, color = result)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("#dddddd", "#aaaaff", "blue", "red")) +
  labs(title = "Errors in prediction",
       subtitle = "Observations on profMu over dmMu plot.",
       color = "Classification result")

profMuDmMuPlot # preview

saveRDS(profMuDmMuPlot, file = "profMuDmMuPlot.Rds")

profSigmaDmSigmaPlot <- starsAndErrors %>% 
  arrange(result) %>%
  ggplot(aes(x = profSigma, y = dmSigma, color = result)) +
  geom_point(size = 1) +
  scale_color_manual(values = c("#dddddd", "#aaaaff", "blue", "red")) +
  labs(title = "Errors in prediction",
       subtitle = "Observations on profSigma over dmSigma plot.",
       color = "Classification result")

profSigmaDmSigmaPlot # preview

saveRDS(profSigmaDmSigmaPlot, file = "profSigmaDmSigmaPlot.Rds")



# Errors exploration in interactive 3d ----

library(plotly)

levelsColors <- c("#dddddd", "#aaaaff", "red", "blue")

p3d <- plotly::plot_ly(data = starsAndErrors, 
                       x = ~profK, 
                       y = ~dmMu,
                       z = ~profMu,
                       color = ~result,
                       colors = levelsColors,
                       size = 1)

p3d

p3d <- plotly::plot_ly(data = starsAndErrors, 
                       x = ~dmK, 
                       y = ~profMu,
                       z = ~dmMu,
                       color = ~result,
                       colors = levelsColors,
                       size = 1)

p3d





