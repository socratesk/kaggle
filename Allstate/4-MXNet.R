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
library(data.table)
library(MASS)
library(mxnet)
library(dummies)
library(caret)

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
fulldata <- dummy.data.frame(fulldata, names=charcols, sep="_")

# Split train and test data sets
train 	 <- fulldata[fulldata$loss > 0,  ]
test 	 <- fulldata[fulldata$loss == 0, ]

# Normalize data. Add 1 to given value to eliminate zeros if any. Take Log
train$loss 	<- log(train$loss + 1)

# Remove unused objects and features from datasets to relinquish memory
train$id	<- NULL
test$id		<- NULL
test$loss 	<- NULL
fulldata 	<- NULL

# Define number of folds
no_folds 	<- 10

# Create number of folds
folds		<- createFolds(1:nrow(train), k = no_folds)

# Extract total no of features as it will be reused many times
total_cols	<- ncol(train)

# Create Parameter list to be passed to NeuralNet Algorithm
params_nn <- list(
				learning.rate	= 0.0001,
				momentum		= 0.9,
				batch.size		= 100,
				wd 				= 0,
				num.round		= 200
			)

# Create palce-holder list object
predictions<- list()

# Train one net
neuralnet_model <- function(train_fold, valid_fold, params, no_round) {

  # Define input data
  inp <- mx.symbol.Variable('data')
  
  # Specify number of fully connected layers for first hidden layer
  l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 40)	#num.hidden = 400
  
  # Set Activation Type as Rectified Linear Unit
  a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
  d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
  
  # Specify number of fully connected layers for second hidden layer
  l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 20)	#num.hidden = 200
  
  # Set Activation Type as Rectified Linear Unit
  a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
  d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
  
  # Specify number of fully connected layers for the last hidden layer
  # This is the convergence layer that produces final output
  l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 1)
  
  # Use MAE Regression for the output layer
  outp <- mx.symbol.MAERegressionOutput(l3, name = "outp")

  # Create a Model
  m <- mx.model.FeedForward.create(outp, 
						X = data.matrix(t(train_fold[, -total_cols])),
						y = 		      train_fold[,  total_cols],
                                    eval.data =
                                    list(data = data.matrix(t(valid_fold[, -total_cols])),
                                         label = valid_fold[,  total_cols]),
                                   array.layout = 'colmajor',
                                   eval.metric	= mx.metric.mae,
                                   learning.rate= params$learning.rate,
                                   momentum 	= params$momentum,
                                   wd 			= params$wd,
                                   num.round 	= params$num.round,
						ctx=mx.cpu(),
						array.batch.size = params$batch.size
				 )
  
  # Predict "loss" for validation data set
  pred_valid <- exp(t(predict(m, data.matrix(t(valid_fold[, -total_cols])), array.layout = 'colmajor'))) - 1
  
  # Predict "loss" for Train data set
  pred_train <- exp(t(predict(m, data.matrix(t(train_fold[, -total_cols])), array.layout = 'colmajor'))) - 1
  
  # Predict "loss" for Test data set
  pred_test  <- exp(t(predict(m, data.matrix(t(test)), array.layout = 'colmajor'))) - 1
  
  # Compute MAE for validation data set
  MAE_valid <- mean(abs(pred_valid - valid_fold[, total_cols]))
  
  # Compute MAE for Train data set
  MAE_train <- mean(abs(pred_train - train_fold[, total_cols]))
  
  # Print MAE values
  print(paste0("Model ", no_round, ". MAE Validation: ", MAE_valid, ". MAE Train: ", MAE_train, ". \n"))
  
  return(pred_test)
}

# Create 10 fold model and predict "loss" values 
for (i in 1:no_folds) {
  predictions[[i]] <- neuralnet_model(
						train_fold = train[-folds[i][[1]], ],
						valid_fold = train[ folds[i][[1]], ],
						params = params_nn,
						no_round = i
                      )
}

# Create Submission file for single model
submission 		<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- predictions[1]
write.csv(submission, "Submission-4-MxNet-Single.csv", row.names = FALSE, quote=FALSE)

# Create Submission file for average of all the models
submission 		<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- apply(predictions, 1, function(x) { mean(x, na.rm=TRUE) })
write.csv(submission, "Submission-4-MxNet-Mean.csv", row.names = FALSE, quote=FALSE)

