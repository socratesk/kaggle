###############################
#                             #
#  TELSTRA TELECOMMUNICATIONS #
#                             #
###############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/telstra-recruiting-network/

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/home/kaggle/Telstra")

# Load required Packages
library(reshape2)
library(xgboost)
library(methods)
library(caret)

# Set seed for reproducibility
set.seed(220469)

# Load Train dataset and rearrange 'Location' feature
len <- nchar('location ')
train 			<- read.csv('train.csv',header=TRUE,stringsAsFactors = F)
train$location 	<- substr(train$location, len + 1, len + 4)
train$location 	<- as.numeric(gsub(' ', '', train$location))

# Load Test dataset and rearrange 'Location' feature
test 			<- read.csv('test.csv',header=TRUE,stringsAsFactors = F) # 11171 x 2
test$location 	<- substr(test$location, len + 1, len + 4)
test$location 	<- as.numeric(gsub(' ', '', test$location))

# Load and expand 'Resource Type' feature
len 					<- nchar('resource_type ')
resource 				<- read.csv('resource_type.csv', header=TRUE, stringsAsFactors = F)
resource$value 			<- as.numeric(substr(resource$resource_type, len + 1, len + 2))
resource$resource_type 	<- gsub(' ', '.', resource$resource_type)
resource_df 			<- dcast(resource, id ~ resource_type, value.var = "value")

# Replace NAs to -1 in 'Resource Type' feature
resource_df[is.na(resource_df)] <- 0

# Load and expand 'Event Type' feature
len 				<- nchar('event_type ')
event 				<- read.csv('event_type.csv', header=TRUE, stringsAsFactors = F)
event$value 		<- as.numeric(substr(event$event_type, len + 1, len + 2))
event$event_type 	<- gsub(' ', '.', event$event_type)
event_df 			<- dcast(event, id ~ event_type, value.var = "value")

# Replace NAs to -1 in 'EventType' feature
event_df[is.na(event_df)] <- 0

# Load and expand 'Severity Type' feature 
len 					<- nchar('severity_type ')
severity 				<- read.csv('severity_type.csv', header=TRUE, stringsAsFactors = F) # 18552 x 2
severity$value 			<- as.numeric(substr(severity$severity_type, len + 1, len + 2))
severity$severity_type 	<- gsub(' ', '.', severity$severity_type)
severity_df 			<- dcast(severity, id ~ severity_type, value.var = "value") # 18552 x 6

# Replace NAs to -1 in 'EventType' feature
severity_df[is.na(severity_df)] <- 0

# Merge Train data with 'Resource Type', 'Event Type', 'Severity' data
train_res 			<- merge(train, resource_df, by="id")
train_res_evn 		<- merge(train_res, event_df, by="id")
train_res_evn_sev 	<- merge(train_res_evn, severity_df, by="id")
trainold 			<- train
train 				<- train_res_evn_sev
rm(train_res_evn_sev)

# Merge Test data with 'Resource Type', 'Event Type', 'Severity' data
test_res 			<- merge(test, resource_df, by="id")
test_res_evn 		<- merge(test_res, event_df, by="id")
test_res_evn_sev 	<- merge(test_res_evn, severity_df, by="id")
testold 			<- test
test 				<- test_res_evn_sev
rm(test_res_evn_sev)

# Remove ID fields
testid 		<- test$id 
train$id 	<- NULL
test$id 	<- NULL

# Remove unused 'Event Type' features
train$event_type.17 <- NULL
train$event_type.33 <- NULL
train$event_type.4 <- NULL
train$event_type.48 <- NULL
train$event_type.52 <- NULL

test$event_type.17 <- NULL
test$event_type.33 <- NULL
test$event_type.4 <- NULL
test$event_type.48 <- NULL
test$event_type.52 <- NULL

# Determine Predictor
predictorname 	<- setdiff(names(train), names(test))
predictor 		<- train[, predictorname]

# Determine no of classes in Predictor feature
noOfClasses 	<- length(unique(train[, predictorname]))

# Set no of rounds
cv.round 		<- 2500

# Set no of Cross Validation folds
cv.nfold 		<- 5

# Remove Location feature
train$location 	<- NULL
test$location 	<- NULL

# Extract 10% of data from Train data set for cross-validation and to monitor the performance of model
trainIndex 		<- createDataPartition(train$location, p=0.1, list=FALSE) 
dtest 			<- train[ trainIndex, ]
train 			<- train[-trainIndex, ]

# Extract Predictor from dtest data set
dtestpredictor 	<- dtest[, predictorname]

# Extract Predictor from train data set
trainpredictor 	<- train[, predictorname]

# Remove predictors from Train and validation data set it got extracted already
dtest[, predictorname] <- NULL
train[, predictorname] <- NULL

# Create DMatrix object from dtest data set - Will be used for watch list. 
dtestmatrix <- xgb.DMatrix(data = data.matrix(dtest), label = dtestpredictor) 

# Create XGB Matrix
trainmatrix <- xgb.DMatrix(data=data.matrix(train), label = trainpredictor)

# Create watchlist
watchlist <- list(train = trainmatrix, test = dtestmatrix, verbose=1)

# Prepare parameter list
param <- list(objective = 'multi:softprob'
		  , eval_metric = 'mlogloss'
		  , booster 	= "gbtree"
		  , num_class 	= noOfClasses
		  , max.depth 	= 6
		  , eta 		= 0.023
		  , subsamp 	= 0.5
		  , watchlist 	= watchlist
		  , colsample_bytree = 0.5
		)

# XGB Cross Validation
xgbcv <- xgb.cv(param 	= param
			, data 		= trainmatrix
			, label 	= trainpredictor 
			, nrounds 	= cv.round 
			, nfold 	= cv.nfold
		)

# Plot and verify the Global minimum for no of rounds
plot(xgbcv$test.mlogloss.mean, type='l')

# Choose best round when log-loss is minimum
nrounds <- which(xgbcv$test.mlogloss.mean == min(xgbcv$test.mlogloss.mean) )

# Train XGB Model on Train datase with best round chosen above
bst <- xgb.train(param 	= param
			, data 		= trainmatrix
			, label 	= trainpredictor
			, verbose 	= 2
			, nrounds 	= nrounds
			)

# Create Matrix for Test dataset
testMatrix  <- as.matrix(test)

# Predict outcome from XGBModel for Test dataset
ypred 		<- predict(bst, testMatrix)
predMatrix 	<- data.frame(matrix(ypred, byrow = TRUE, ncol = noOfClasses))

# Create Submission file
colnames(predMatrix) <- paste("predict_", c(0,1,2), sep="")
resp 				 <- data.frame(id = testid, predMatrix)
write.csv(resp, 'Submission-1-XGBoost.csv', row.names=F, quote = F)
