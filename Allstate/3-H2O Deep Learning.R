###############################
#                             #
#  ALLSTATE INSURANCE COMPANY #
#                             #
###############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/allstate-claims-severity

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/home/kaggle/AllState")

# Load required Packages
library(h2o)
library(readr)
library(caret)

# Function that calculates Mean Absolute Error
calculateMAE <- function(actual, predicted) {
	mae <- sum(abs(actual - predicted)) / length(actual)
	mae
}

# Set seed for reproducibility
set.seed(456)

# Load Data files
train <- read.csv("train.csv", stringsAsFactor = TRUE)
test  <- read.csv("test.csv", stringsAsFactor = TRUE)

# Normalize data. Add 1 to given value to eliminate zeros if any. Take Log.
train$loss <- log(train$loss + 1)

# Split train data to have validation dataset using createDataPartition (caret package) 
trainindex <- createDataPartition(train$id, p=0.8, list=FALSE)
train80 <- train[ trainindex, ]
valid20 <- train[-trainindex, ]

# Extract actual loss from Validation dataset for comparison
valid20_loss	<- exp(valid20$loss)-1

# Remove unused features from datasets
train80$id 		<- NULL
valid20$id		<- NULL
test$id			<- NULL

# Extract total no of features as it will be reused many times.
total_cols 		<- ncol(train80)

# Start H2O processes
h2o.init(nthreads = -1, max_mem_size = "16g") # -1: Use all CPUs on host

# Create H2O dataframe for all data sets
train80_h2o 	<- as.h2o(train80)
valid20_h2o 	<- as.h2o(valid20)
valid20_pred_h2o<- as.h2o(valid20[, -total_cols])
test_h2o 	<- as.h2o(test)

## Create H2O DeepLearning model. Function created for repeatable use.
train_h2o_nn_model <- function(epochs, hidden_layer, distribution, activation, run) {
	h2o_nn_model <- h2o.deeplearning(
				x		= 1:total_cols-1, 
				y		= total_cols, 
				training_frame	= train80_h2o, 
				validation_frame= valid20_h2o,
				epochs		= epochs,	
				stopping_rounds	= 2,
				activation	= activation,
				distribution	= distribution,
				hidden		= hidden_layer,
				overwrite_with_best_model = T
			   )
	
	# Apply the model on validation dataset to compare with actuals
	valid20_y_hat <- as.matrix(predict(h2o_nn_model, valid20_pred_h2o))  #
	valid20_y_hat <- exp(valid20_y_hat) - 1
	
	# Calculate MAE on validation dataset and print
	print(paste0("MAE ", run, ": ", calculateMAE(valid20_loss, valid20_y_hat)))

	# Apply the model on test dataset and Predict the loss outcome
	y_hat <- exp(predict(h2o_nn_model, test_h2o)) - 1

	y_hat
}

# Create multiple DeepLearning models and calculate predictions for test dataset
y_hat1 <- as.matrix(train_h2o_nn_model(20, c(100,100), "huber", "Rectifier", 1))
y_hat2 <- as.matrix(train_h2o_nn_model(20, c(120,100), "huber", "Rectifier", 2))
y_hat3 <- as.matrix(train_h2o_nn_model(20, c(120,120), "huber", "Rectifier", 3))
y_hat4 <- as.matrix(train_h2o_nn_model(20, c(80,80,80), "huber", "Rectifier", 4))
y_hat5 <- as.matrix(train_h2o_nn_model(20, c(91,91), "huber", "Rectifier", 5))

# Take simple average (Ensemble) of all the predictions 
y_hat_pred <- (y_hat1 + y_hat2 + y_hat3 + y_hat4 + y_hat5)/5

# Create Submission file
submission 	<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- y_hat_pred
write.csv(submission, "Submission_3_H2O_DeepLearn-Ensemble", row.names = FALSE)
