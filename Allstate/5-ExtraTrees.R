###############################
#                             #
#  ALLSTATE INSURANCE COMPANY #
#                             #
###############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally
# COMPETITION SITE URL: https://www.kaggle.com/c/allstate-claims-severity

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/home/kaggle/AllState") 

# Load required Packages
library(dummies)
library(caret)

# Set memory before calling Extra Tree Library
options( java.parameters = "-Xmx16g" )

# Load Extra Tree Libray
library(extraTrees)

# Function that calculates Mean Absolute Error
calculateMAE <- function(actual, predicted) {
	mae <- sum(abs(actual - predicted)) / length(actual)
	mae
}

# Set seed for reproducibility
set.seed(456)

# Load Data files
traindata <- read.csv("train.csv", stringsAsFactor = TRUE)
testdata  <- read.csv("test.csv",  stringsAsFactor = TRUE)

# Introduce loss feature in Test data set
testdata$loss <- 0

# Combine both Test and Train data sets
fulldata <- rbind(traindata, testdata)

# Prepare list of Categorical feature names
trainColNames <- paste0(c('cat'), 1:116)

# Remove unused objects to relinquish memory
traindata <- NULL
testdata  <- NULL

### Some Feature engineering
# Collect Categorical variables that have more than two categories
charcols <- c()
for (colnm in trainColNames) {
  if (class(colnm) == 'character') {
    if (length(unique(fulldata[[colnm]])) == 2) {
    	fulldata[[colnm]] <- as.numeric(as.factor(fulldata[[colnm]]))
	fulldata[[colnm]] <- fulldata[[colnm]] - 1
    } else {
	charcols <- c(charcols, colnm)
    }
  }
}

# Perform One hot-encoding
full <- dummy.data.frame(fulldata, names=charcols, sep="_")

# Remove fulldata from memory
fulldata <- NULL

# Split train and test data sets
train 	 <- full[full$loss > 0,  ]
test 	 <- full[full$loss == 0, ]

# Normalize data. Add 1 to given value to eliminate zeros if any. Take Log
train$loss <- log(train$loss + 1)

# Remove unused objects and features from datasets to relinquish memory
test$id		<- NULL
test$loss 	<- NULL
full 		<- NULL
charcols	<- NULL
trainColNames 	<- NULL

# Split train data to have validation dataset using createDataPartition (caret package) 
trainindex 	<- createDataPartition(train$id, p=0.85, list=FALSE)
train85 	<- train[ trainindex, ]
valid15 	<- train[-trainindex, ]

# Remove ID from train data set
train85$id	<- NULL
valid15$id	<- NULL
train		<- NULL
trainindex  	<- NULL

# Define parameters
nodesize 	<- 6
numRandomCuts 	<- 3
numThreads	<- 4
ntree		<- 800

# Extract total no of features as it will be reused many times
total_cols	<- ncol(train85)

# Create palce-holder list object
predictions	<- list()

# Create Extra Trees model and predict Test data set's "loss" value
extraTrees_model <- function(node, numRandCuts, numThreads, ntree) {

  # Model creation
  m <- extraTrees(x		= train85[, -total_cols],
		  y 		= train85[,  total_cols],
		  nodesize 	= node,
		  numRandomCuts = numRandCuts,
		  numThreads	= numThreads,
		  ntree		= ntree	
		)
  
  # Predict "loss" for validation data set
  pred_valid <- exp(predict(m, valid15[, -total_cols])) - 1
  
  # Predict "loss" for Train data set
  pred_train <- exp(predict(m, train85[, -total_cols])) - 1
  
  # Predict "loss" for Test data set
  pred_test  <- exp(predict(m, test)) - 1
  
  # Compute MAE for validation data set
  MAE_valid <- calculateMAE(pred_valid, valid15[, total_cols])
  
  # Compute MAE for Train data set
  MAE_train <- calculateMAE(pred_train, train85[, total_cols])
  
  # Print MAE values
  print(paste0("Model ", numRandCuts, ". MAE Validation: ", MAE_valid, ". MAE Train: ", MAE_train, ". \n"))
  
  return(pred_test)
}

# Create 1, 2, and 3 cut models and predict "loss" values 
for (i in 1:numRandomCuts) {
	predictions[[i]] <- extraTrees_model(nodesize, i, numThreads, ntree)
}

# Create Submission file for single model
submission 	<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- predictions[1]
write.csv(submission, "Submission-5-ExtraTree-ThirdTree.csv", row.names = FALSE, quote=FALSE)

# Create Submission file for average of all the models
submission 	<- read.csv("submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- apply(predictions, 1, function(x) { mean(x, na.rm=TRUE) })
write.csv(submission, "Submission-4-ExtraTree-Mean.csv", row.names = FALSE, quote=FALSE)
