################################
## PropertyPrice Project
################################

################################
# Installing packages/libraries
################################

# Installing required packages and libraries
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(GGally)) install.packages("GGally", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
###if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")

library(lubridate)
library(psych)
library(corrplot)
library(broom)
library(randomForest)

# Preventing scientific notation  
options(scipen=999)

################################
# Downloading data
################################
#
# Source data set can be downloaded at kaggle.com
# https://www.kaggle.com/harlfoxem/housesalesprediction
# You also can find it at: 
#     https://github.com/galindo-raul/PropertyPrice/blob/master/data/kc_house_data.csv

# For this project, we have downloaded the data set into ./data/kc_house_data.csv
# Reading data set into a df
data <- read.csv("data/kc_house_data.csv")

# feature names
names(data)

# Features description
###############################
# id            : Unique ID for each observation (home sold)
# date          : Date of the home sale
# price         : Price of each home sold
# bedrooms      : Number of bedrooms
# bathrooms     : Number of bathrooms, where .5 accounts for a room with a toilet but no shower
# sqft_living   : Square footage of the apartments interior living space
# sqft_lot      : Square footage of the land space
# floors        : Number of floors
# waterfront    : A dummy variable for whether the apartment was overlooking the waterfront or not
# view          : An index from 0 to 4 of how good the view of the property was
# condition     : An index from 1 to 5 on the condition of the apartment
# grade         : An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design
# sqft_above    : The square footage of the interior housing space that is above ground level
# sqft_basement : The square footage of the interior housing space that is below ground level
# yr_built      : The year the house was initially built
# yr_renovated  : The year of the house's last renovation
# zipcode       : Zipcode area
# lat           : Lattitude
# long          : Longitude
# sqft_living15 : The square footage of interior housing living space for the nearest 15 neighbors
# sqft_lot15    : The square footage of the land lots of the nearest 15 neighbors
#
################################
# Data Observation
################################
#
glimpse(data)
summary(data)

# Checking if there is NA's
sapply(data, function(x) sum(is.na(x)))

# there is only one year data so we can't consider that we have historical data
range(ymd(substring(data$date,1,8)))

# let's check houses with 0 or 33 bedrooms or 0 bathrooms 
data %>% filter( bedrooms %in% c(0,33) | bathrooms == 0 ) %>% as.tibble()
# We will considere houses with zero or 33 bedrooms as outliers
# We will considere houses with zero bathrooms as outliers
# Removing outliers
data <- data %>% filter(!(bedrooms  %in% c(0,33)) 
                        & bathrooms != 0 ) 

################################
# Data Wrangling
################################
# 
# date feature follows the pattern yyyymmddT000000 where yyyy stands for year, mm stands for month and dd stands for day, 
# let's simplify the date of the sale as yyyymm with numeric class 
data <- mutate (data, date = as.numeric(substring(data$date,1,6))) 

################################
# Preprocesising Data
################################
# 
# Identify features with low variability
nzv <- nearZeroVar(data)

#names_nzv <- names(data)[nzv]

# Removing id, price and nzv features
colsRemoved <- c(1, 3, nzv)

# Predictors
col_index <- setdiff(1:ncol(data), colsRemoved)

# Let´s see how Price is correlated with other variables
ggcorr(data[,c(3,col_index)], 
       name = "corr", 
       label = TRUE, 
       hjust = 1, 
       label_size = 2.5, 
       angle = -45, 
       size = 3)

# Removing sqft_above
colsRemoved <- c(13,colsRemoved)
col_index <- setdiff(1:ncol(data), colsRemoved)

# Let's see the predictors set
names(data)[col_index]

################################
# Spliting dataset
################################
#
#Setting the seed of R‘s random number generator in order to be able to reproduce the random objects like test_index
set.seed(1, sample.kind="Rounding")

# createDataPartition: generates indexes for randomly splitting the data into training and test sets
# test set will be around 20% of the dataset, we pick p a little over 0.20 as we will remove 
# from test_set any observation with the choosen features which not exists on the train_set 
test_index <- createDataPartition(y = data$price, times = 1, p = 0.10, list = FALSE)

train_set <- data[-test_index,]
test_set <- data[test_index,]

# checking the proportion of the test_set
nrow(test_set) / nrow(data) * 100

#
################################
# Visualization
################################
#
# Let's see price distribution in the train data set
# we use log10 transformation on the x-axis
train_set %>%
  ggplot(aes(price)) +
  geom_histogram(fill = "blue", color = "black") + 
  scale_x_continuous(trans = "log10") +
  xlab("log10(price)")

# Let's check price vs sqft_living by using linear regresion
train_set %>%
  ggplot(aes(sqft_living, price)) +
  geom_point(alpha = .25, color = "blue") + 
  geom_smooth(method = "lm",color="red")

# Let's see sqft_living distribution in the train data set
# we use log10 transformation on the y-axis
train_set %>%
  ggplot(aes(sqft_living)) +
  geom_histogram(fill = "blue", color = "black",binwidth = 500) + 
  scale_y_continuous(trans = "log10")  +
  ylab("log10(count)")

# grade
# Let's check price vs grade by using box plot
# We will not consider grade = 3 since there is only one observation
# we use log10 transformation on the y-axis
train_set %>% filter(!(grade %in% c(3, 13))) %>%
  ggplot(aes(as.factor(grade), price)) +
  geom_boxplot(fill = "blue", color = "black") + 
  scale_y_continuous(trans = "log10") +
  xlab("grade")  +
  ylab("log10(price)")

# Let's see grade distribution in the train data set
train_set %>%
  ggplot(aes(as.factor(grade))) +
  geom_bar(fill = "blue", color = "black") + 
  xlab("grade")

# Let's check price vs sqft_living15 by using linear regresion
# we use log10 transformation on both axes
train_set %>%
  ggplot(aes(sqft_living15, price)) +
  geom_point(alpha = .25, color = "blue") + 
  geom_smooth(method = "lm",color="red")

# Let's see sqft_living15 distribution in the train data set
# we use log10 transformation on the y-axis
train_set %>%
  ggplot(aes(sqft_living15)) +
  geom_histogram(fill = "blue", color = "black",binwidth = 300) + 
  scale_y_continuous(trans = "log10") + 
  ylab("log10(count)")

# let's take bathrooms as categorical data
# Let's check bathrooms vs grade by using box plot
# removing groups with less than 10 houses
# we use log10 transformation on the y-axis
train_set %>%
  group_by(bathrooms) %>%
  filter(n() >= 50) %>% ungroup() %>%
  ggplot(aes(as.factor(bathrooms), price)) +
  geom_boxplot(fill = "blue", color = "black") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_y_continuous(trans = "log10") +
  xlab("bathrooms") + 
  ylab("log10(price)")

# bathrooms distribution
train_set %>%
  ggplot(aes(bathrooms)) +
  geom_bar(color="black",fill="blue") + 
  scale_y_continuous(trans = "log10") +
  xlab("bathrooms")

################################
# Machine Learning Techniques
################################

# House price predictions will be compared to the true prices in the test_set using RMSE
# Creating the function RMSE that computes the RMSE 
# our goal is to build an algorithm that minimizes RMSE
#
RMSE <- function(true_prices, predicted_prices){
  sqrt(mean((true_prices - predicted_prices)^2))
}

# Last Square Estimate (LSE) 
#===========================================

# start time
t1 <- Sys.time()

fit <- lm(price ~ ., data = train_set)

# end time
t2 <- Sys.time()

# computation time
compTime <- t2 - t1

# Let's check out all the information provided by lm function
summary(fit)

tidy(fit, conf.int = TRUE)

# Predicting prices on test set
predicted_prices1 <- predict(fit, test_set)

# Comparing predicted prices vs actual prices
rmse1 <- RMSE(test_set$price, predicted_prices1)

# Storing the result
rmse_results <- data_frame(model = 1,
                           method = "Last Square Estimate LSE",
                           RMSE = rmse1,
                           improvement = 0,
                           time = round(compTime,2))

# Checking out results
rmse_results %>% knitr::kable()

# Model 2: knn - k-nearest neighbors with cross validation
#===========================================================

# We set the seed because cross validation is a random procedure and we want to make sure the result here is reproducible.
set.seed(2000)

# Cross validation trying out values between 5 and 61
# It takes around 27 minutes in my laptop
# start time
t1 <- Sys.time()

train_knn <- train(train_set[,col_index], train_set[,"price"],
                   method = "knn", 
                   tuneGrid = data.frame(k = seq(7,61,2))
                   )

# end time
t2 <- Sys.time()

# computation time
t2 - t1

# Visualizing RMSE's
ggplot(train_knn, highlight = TRUE)

# The parameter that minimizes RMSE
train_knn$bestTune

# The best performing model
train_knn$finalModel

# Predicting prices on test set
predicted_prices2 <- predict(train_knn, test_set)

# Comparing predicted prices vs actual prices
rmse2 <- RMSE(test_set$price, predicted_prices2)

# Storing the result
rmse_results <- bind_rows(rmse_results,
                          data_frame(model = 2,
                                     method = paste(train_knn$finalModel$k,
                                                    "-nearest neighbor regression model", 
                                                    sep = ""),
                                     RMSE = rmse2,
                                     improvement = round(100*(rmse2/rmse1 - 1),2),
                                     time = round(compTime,2)))

# Checking out results
rmse_results %>% knitr::kable()

# Model 3: gamLoess method
#============================================

# Setting the seed
set.seed(2000)

# We stick degree = 1 and try ten values for span between .15 and .95
grid <- expand.grid(span = seq(0.15, 0.95, len = 10), degree = 1)

# it can takes 30 minutes !!! 
# start time
t1 <- Sys.time()

train_loess <- train(train_set[,col_index], train_set[,"price"],
                     method = "gamLoess",
                     tuneGrid=grid
                     ) 

# end time
t2 <- Sys.time()

# computation time
compTime <- t2 - t1

# Visualizing RMSE's
ggplot(train_loess, highlight = TRUE)

# cross validation results
train_loess$results

# The parameter that minimizes RMSE
train_loess$bestTune

# Predicting prices on test set
predicted_prices3 <- predict(train_loess, test_set)

# Comparing predicted prices vs actual prices
rmse3 <- RMSE(test_set$price, predicted_prices3)

# Storing the result
rmse_results <- bind_rows(rmse_results,
                          data_frame(model = 3,
                                     method = "gamLoess",
                                     RMSE = rmse3,
                                     improvement = round(100*(rmse3/rmse1 - 1),2),
                                     time = round(compTime,2)))

# Checking out results
rmse_results %>% knitr::kable()

# Model 4 : Regression Tree - rpart
#===============================================

# Setting the seed
set.seed(2000)

# We stick degree = 1 and try ten values for span between .15 and .95
# takes 30 secs
# start time
t1 <- Sys.time()

train_rpart <- train(train_set[,col_index], train_set[,"price"],
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25))
                     )

# end time
t2 <- Sys.time()

# computation time
compTime <- t2 - t1

# Visualizing RMSE's
ggplot(train_rpart, highlight = TRUE)

# Predicting prices on test set
predicted_prices4 <- predict(train_rpart, test_set)

# Comparing predicted prices vs actual prices
rmse4 <- RMSE(test_set$price, predicted_prices4)

# Storing the result
rmse_results <- bind_rows(rmse_results,
                          data_frame(model = 4,
                                     method = "Regression Tree - rpart",
                                     RMSE = rmse4,
                                     improvement = round(100*(rmse4/rmse1 - 1),2),
                                     time = round(compTime,2)))

# Checking out results
rmse_results %>% knitr::kable()

#
# Model 5: Regression Tree - randomForest
#==========================================

# Setting the seed
set.seed(2000)
# it takes 3 min
# start time
t1 <- Sys.time()

train_rF <- randomForest(train_set[,col_index], train_set[,"price"])

# end time
t2 <- Sys.time()

# computation time
compTime <- t2 - t1

# Visualizing error
plot(train_rF)
train_rF
# Predicting prices on test set
predicted_prices5 <- predict(train_rF, test_set)

# Comparing predicted prices vs actual prices
rmse5 <- RMSE(test_set$price, predicted_prices5)

# Storing the result
rmse_results <- bind_rows(rmse_results,
                          data_frame(model = 5,
                                     method = "Regression Tree - randomForest",
                                     RMSE = rmse5,
                                     improvement = round(100*(rmse5/rmse1 - 1),2),
                                     time = round(compTime,2)))

# Checking out results
rmse_results %>% knitr::kable()

################################
# Results Analysis
################################

# rmse_results ordered
rmse_results[order(rmse_results$RMSE),] %>% knitr::kable()

# a data frame with error analysis from randomForest  
df <- data_frame(y = test_set$price, 
                 y_hat = predicted_prices5,
                 diffPrice = y - y_hat)

# summary of df
df %>% summarize(avg_y = mean(y),
                 avg_y_hat = mean(y_hat),
                 minError = mean(min(abs(diffPrice))),
                 maxError = mean(max(abs(diffPrice))),
                 avgError = mean(diffPrice),
                 sdError = sd(diffPrice)
)
# error histogram
df %>%
  ggplot(aes(diffPrice)) +
  geom_histogram(bins = 50, fill = "blue", color = "black") +
  scale_y_continuous(trans = "log10") +
  xlab("actual price - predicted price") + 
  ylab("log10(count)")
  
# QQ-plot using standard units
p <- seq(0.05, 0.95, 0.05)
z <- scale(predicted_prices5)
sample_quantiles <- quantile(z, p)
theoretical_quantiles <- qnorm(p) 
qplot(theoretical_quantiles, sample_quantiles) + geom_abline()

