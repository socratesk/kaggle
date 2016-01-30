
#########################################
#                                       #
#   WALMART -TRIP TYPE CLASSIFICATION   #
#                                       #
#########################################
# PRIVATE LB Score: 2.08247 (with 10 rounds); Rank 641/1047

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/walmart-recruiting-trip-type-classification

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/courses/kaggle/Walmart")

# Load Packages required for xgboost
library(xgboost)
library(methods)
library(reshape2)
library(caret)

# Set seed for reproducibility
set.seed(23456)

# Read train and test data set
train <- read.csv('train.csv', header = T, stringsAsFactors = FALSE)
test <- read.csv('test.csv', header = T, stringsAsFactors = FALSE)


### Exploratory Data Analysis and Data Clensing
# Identify missing entries between train and test data sets and remove them
setdiff(unique(train$DepartmentDescription), unique(test$DepartmentDescription))

# Remove the Department that is NOT existing in Test data set
train <- train[! train$DepartmentDescription == "HEALTH AND BEAUTY AIDS", ]

# Impute 1 to missing values [NA] for Upc feature
train[is.na(train$Upc), ]$Upc <- 1	# Train - 4129
test [is.na(test$Upc) , ]$Upc <- 1	# Test - 3986

# Impute 1 to missing values [NA] for FinelineNumber feature
train[is.na(train$FinelineNumber), ]$FinelineNumber <- 1	# Train
test [is.na(test$FinelineNumber) , ]$FinelineNumber <- 1	# Test

# Apply 'log' function to Upc and FinelineNumber features to reduce it dimensions [value spread]
train$Upc <- log(train$Upc)
train$FinelineNumber <- log(train$FinelineNumber)

test$Upc <- log(test$Upc)
test$FinelineNumber <- log(test$FinelineNumber)

# Factorize Weekday 
train$Weekday <- as.numeric(as.factor(train$Weekday)) # Train
test$Weekday  <- as.numeric(as.factor(test$Weekday))  # Test

# Factorize DepartmentDescription
train$DepartmentDescription <- as.numeric(as.factor(train$DepartmentDescription)) # Train
test$DepartmentDescription  <- as.numeric(as.factor(test$DepartmentDescription))  # Test

# Extract unique Target Trip Types
TargetType <- unique(train$TripType)

# Create a mapping data-frame for actual TripType with "zero based" sequential ones
targetdf <- data.frame(target = sort(TargetType), seqno = c(0 : (length(TargetType) -1)) )

# Change the Trip Type to 0 based sequential number instead of random using 'targetdf' dataframe
# Introduce a new variable called 'Target'
train$Target <- 0

# Populate new variable with 'corresponding ordered sequence number' from 'targetdf'.
for (type in TargetType) {
	train[train$TripType == type,]$Target <- targetdf[targetdf$target == type,]$seq
}

# Extract new "zero based" sequential Target
Target <- train$Target

# Determine Number of Classes to be predicted - Will be used for model generation.
noOfClasses <- length(unique(Target))

# Remove Target and TripType features from train data set
train$Target <- NULL
train$TripType <- NULL


### Prep-work for model generation
# Create dummy variables for train dataset's Factor variables
trainDummy <- dummyVars(" ~ .", data = train, fullRank = F)
train <- as.data.frame(predict(trainDummy, train))

# Create dummy variables for test dataset's Factor variables
testDummy <- dummyVars(" ~ .", data = test,  fullRank = F)
test <- as.data.frame(predict(testDummy,  test))

# Create DMatrix object from train data set - Will be used for model generation.
trainMatrix <- xgb.DMatrix(data = data.matrix(train), label = Target)

# Prepare parameter list - - Will be used for model generation.
# Add more parameters for fine tuning or for early stopping.
param <- list('objective' = 'multi:softprob',
		  'eval_metric' = 'mlogloss',
		  'num_class' = noOfClasses)

# Initialise number of rounds and folds. Try with a bigger number initially, and perform cross-validation.
# This is required to identify 'Global-minimum' instead of getting stuck with 'Local-minimum'
cv.round <- 200
cv.nfold <- 5


### Model Generation
# Perform Cross-validation using the above params and objects
xgbcv <- xgb.cv(param = param, data = trainMatrix,
			label = Target, nrounds = cv.round, 
			nfold = cv.nfold)

# Plot to visualise how the cross-validation is performing
# plot(xgbcv$test.mlogloss.mean, type='l')

# Determine 'Global-minimum' number of rounds required for the model
nround <- which(xgbcv$test.mlogloss.mean == min(xgbcv$test.mlogloss.mean) )

# Develop a model using XGBoost and the above params / objects 
xgb_model <- xgboost(param = param, data = trainMatrix, label = Target, nrounds = cv.round)


### Prediction
# Create Dense matrix  from test data set - Will be used for prediction.
testMatrix <- as.matrix(test)

# Predict the value
ypred <- predict(xgb_model, testMatrix)

# Convert predicted values into Matrix as stated in sample-submission
predMatrix <- data.frame(matrix(ypred, byrow = TRUE, ncol = noOfClasses))


### Output
# Create column header for Output file
colnames(predMatrix) <- paste("TripType_", targetdf$target, sep="")

# Combine column header and predicted values as data frame
res <- data.frame(VisitNumber = test[, 1], predMatrix)

# Perfrom aggregation on Visit number by taking 'average'
result <- aggregate(. ~ VisitNumber, data = res, FUN = mean)

# Write to Output file
write.csv(format(result, scientific = FALSE), '1-XGBoost.csv', row.names = F, quote = F) # Submission-5
