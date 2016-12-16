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
library(gbm)
library(caret)

# Function that calculates Mean Absolute Error
calculateMAE <- function(actual, predicted) {
	mae <- sum(abs(actual - predicted)) / length(actual)
	mae
}

# Set seed for reproducibility
set.seed(456)

# Load Data files
train <- read.csv("train.csv", stringsAsFactor = FALSE)
test  <- read.csv("test.csv", stringsAsFactor = FALSE)

# Normalize data. Add 1 to given value to eliminate zeros if any. Take Log. Finally compute 1/x
train$loss <- 1/log(train$loss + 1)

# Assign all the features to a variable
features = names(train)
 
# Convert character features to numerical factor features
for (f in features) {
  if (class(train[[f]])=="character") {
    levels <- sort(unique(train[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
  }
}

# Split train data to have validation dataset using createDataPartition (caret package) 
trainindex <- createDataPartition(train$id, p=0.8, list=FALSE)
train80 <- train[ trainindex, ]
valid20 <- train[-trainindex, ]

# Extract actual loss from Validation dataset for future comparison
valid20_loss	<- exp(1/valid20$loss)-1

# Remove unused features from datasets
train80$id 		<- NULL
valid20$id		<- NULL
valid20$loss	<- NULL
test$id			<- NULL

# Set no of trees
numtrees <- 1600

# Train Generalised Boosted Regression Model (GBM)
gbm_model <- gbm(formula 		= loss ~ .,
				distribution 	= "gaussian",
				data 			= train80,
				n.trees 		= numtrees,
				n.cores 		= 4,
				n.minobsinnode 	= 10,		# Minm observations in node
				shrinkage 		= 0.002,	# Learning Rate
				bag.fraction 	= 0.5,
				train.fraction 	= 0.8,
				cv.folds 		= 5,		# 5 Fold Cross-Validation
				keep.data 		= TRUE,
				verbose 		= TRUE,
				class.stratify.cv = NULL,
				interaction.depth = 1		# 1: Additive, 2: 2-way interaxns
				)

# Choose best iteration from trained model 
best_iter <- gbm.perf(gbm_model, method="OOB") 	# Perf check using Out-Of-Bag estimator

# Apply the model on validation dataset and Predict the loss outcome
valid20_y_hat 	<- predict(gbm_model, newdata=valid20, best_iter)
valid20_y_hat	<- exp(1/valid20_y_hat) - 1

# Calculate MAE on validation dataset
MAE <- calculateMAE(valid20_loss, valid20_y_hat)

# Print calculated MAE
print(paste0("MAE: ", MAE))

# Apply the model on test dataset and Predict the loss outcome
y_hat <- predict(gbm_model, newdata=test, best_iter)
y_hat <- exp(1/y_hat) - 1

# Create Submission file
submission 		<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- y_hat
write.csv(submission, "Submission-2-GBM", row.names = FALSE)
