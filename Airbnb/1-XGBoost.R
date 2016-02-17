##########################################
#                                        #
#  Airbnb - NEW USER BOOKING PREDICTION  #
#                                        #
##########################################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings
# PRIVATE LB SCORE: 0.87004 (would place you to 505 out of 1463)

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/home/kaggle/airbnb")

# Load libraries required for xgboost
library(xgboost)
library(readr)
library(caret)
library(lubridate)
library(car)

# Set seed for reproducibility
set.seed(23456)

# Read train and test data set
train <- read_csv("train_users_2.csv")
test <- read_csv("test_users.csv")

# Extract Response variable from Train data set
Response <- train$country_destination

# Remove the Response variable from Training Data set
train$country_destination <- NULL

# Combine both Train and Test data sets
train_test <- rbind(train, test)


## Do some feature Engineering
# The 'date_first_booking' feature may not have much impact since it talks about past booking. So remove this feature.
train_test$date_first_booking <- NULL

# Replace any missing values in Train and Test data sets to -1
train_test[is.na(train_test)] <- -1

# Convert CHR Date field into actual Date
train_test$date_account_created <- ymd(as.Date(train_test$date_account_created, format="%Y-%m-%d"))

# Extract Day, Month, and Year from Date object and then remove Data object
train_test$DAC_Day   <- day(train_test$date_account_created)
train_test$DAC_Month <- month(train_test$date_account_created)
train_test$DAC_Year  <- year(train_test$date_account_created)

# Remove Account Created Date from data set
train_test$date_account_created  <- NULL

# Remove scientific notation to regular one for Timestamp feature AND convert it into Character field
train_test$timestamp_first_active <- as.character(format(train_test$timestamp_first_active, scientific=FALSE))

# Split Timestamp_first_active in year, month and day
train_test$TFA_Day   <- substring(train_test$timestamp_first_active, 7, 8)
train_test$TFA_Month <- substring(train_test$timestamp_first_active, 5, 6)
train_test$TFA_Year  <- substring(train_test$timestamp_first_active, 1, 4)

# Remove First Active Timestamp from data set
train_test$timestamp_first_active <- NULL

# Clean Age by removing values
train_test$age[train_test$age < 14] <- -1
train_test$age[train_test$age > 100] <- -1


## Creation of Dummy Variables for Category variables
# Capture category variables for which dummies are to be created.
encode_feature_list <- c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser')

# Create Dummy Variables (features) for those category variables from combined data sets.
dummies <- dummyVars(~ gender + signup_method + signup_flow + language + affiliate_channel + affiliate_provider + first_affiliate_tracked + signup_app + first_device_type + first_browser, data = train_test)

# Create Dummy DataFrame
dummies_dataframe <- as.data.frame(predict(dummies, newdata = train_test))

#Combine both dummies features 
train_test_combined <- cbind(train_test[, setdiff(names(train_test), encode_feature_list)], dummies_dataframe)

# split train and test data sets from combined set
train_data <- train_test_combined[train_test_combined$id %in% train$id, ]
test_data  <- train_test_combined[train_test_combined$id %in% test$id,  ]

# Code CHR values into Numeric
Response <- recode(Response,"'NDF'=0; 'US'=1; 'other'=2; 'FR'=3; 'CA'=4; 'GB'=5; 'ES'=6; 'IT'=7; 'PT'=8; 'NL'=9; 'DE'=10; 'AU'=11")

# Remove ID column from train data set
train_data$id <- NULL

# Choose XGBoost parameters
n.round <- 225
max.depth <- 8
num.class <- length(unique(Response)) # No of predictions to be made


## Model Generation
# Develop XGBoost model
xgb <- xgboost(data = data.matrix(train_data), 
               label = Response, 
               eta = 0.02,
               max_depth = max.depth, 
               nround = n.round, 
               subsample = 0.7,
               colsample_bytree = 0.8,
               seed = 1,
               objective = "multi:softprob",
               num_class = num.class,
               nthread = 3
)

# predict values in test set
y_pred <- predict(xgb, data.matrix(test_data[, -1]))

# Format predicted values into dataframe
# NOTE: In this case, dataframe is horizontally spread than vertical one to facilitate sorting
# No of columns is equal to no of users in Test data set.
y_pred <- as.data.frame(matrix(y_pred, nrow = num.class))

# Assign row names to the dataframe
rownames(y_pred) <- c('NDF','US','other','FR','CA','GB','ES','IT','PT','NL','DE','AU')

# The Submission requires top 5 probable countries for each user. To arrive at that,
# a. Apply sort function to sort all the column values 
# b. Take first 5 values (classes) from sorted column
# c. Get corresponding row-names for each column
# d. Convert them into vector to easily attach in dataframe
first_5_result <- as.vector(apply(y_pred, 2, function(x) names(sort(x)[12:8])))

# Repeat each user ID for 5 times 
ids <- NULL
for (i in 1:nrow(test_data)) {
  idx <- test_data$id[i]
  ids <- append(ids, rep(idx, 5))
}

# Create result dataframe for CSV file
result <- data.frame(id = ids, country = first_5_result)
write.csv(result, "submission-1.csv", quote=FALSE, row.names = FALSE)
