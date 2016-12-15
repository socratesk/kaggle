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
library(caret)
library(xgboost)

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

# Normalize data. Add 1 to given value to eliminate zeros if any. Take Log. Finally compute 1/x
train$loss <- 1/log(train$loss + 1)

# Split train data to have validation dataset using createDataPartition (caret package) 
trainindex <- createDataPartition(train$id, p=0.8, list=FALSE)
train80	   <- train[ trainindex, ]
valid20    <- train[-trainindex, ]

# Extract actual loss from Validation dataset for future comparison
valid20_loss	<- exp(1/valid20$loss)-1

# Remove unused features from datasets
train80$id 	<- NULL
valid20$id	<- NULL
valid20$loss	<- NULL
test$id		<- NULL

## XGBoost Algorithm starts here ##
# Create a string for Response variable
Response <- "loss"

# Create XGBoost compatible matrix for XGBoost Algorithm
train80_matrix <- xgb.DMatrix(data=data.matrix(train80[, names(train80) != Response]), label=train80$loss)

# Set values for XGBoost parameters
nfold	<- 6
nrounds	<- 2500

# Create Parameter list to be passed to XGBoost Algorithm
param <- list(	objective 	 = "reg:linear",
		nthread 	 = 4,
		eta       	 = 0.3,
		booster		 = "gbtree",
		max_depth 	 = 10,
		subsample 	 = 0.75,
		colsample_bytree = 0.8,
		min_child_weight = 1.05
	   )

# Develop a model with train dataset
xgbtrain <- xgb.train(	data	= train80_matrix,
			depth   = 10,
			nrounds = nrounds, 
			param 	= param,
			verbose = 1,
			early_stopping_round = 25
          	     )

# Apply the model on validation dataset and Predict the loss outcome
valid20_y_hat 	<- predict(xgbtrain, data.matrix(valid20))
valid20_y_hat	<- exp(1/valid20_y_hat) - 1

# Calculate MAE on validation dataset
MAE <- calculateMAE(valid20_loss, valid20_y_hat)

# Print calculated MAE
print(paste0("MAE: ", MAE))

# Apply the model on test dataset and Predict the loss outcome
y_hat <- predict(xgbtrain, data.matrix(test))
y_hat <- exp(1/y_hat) - 1

# Create Submission file
submission 	<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- y_hat
write.csv(submission, "Submission-1-Basic-XGBoost", row.names = FALSE)
