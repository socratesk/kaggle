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
library(data.table)
library(Metrics)
library(scales)
library(Hmisc)
library(forecast)
library(caret)
library(xgboost)
library(e1071)
library(stringr)

# Function that calculates Mean Absolute Error
calculateMAE <- function(actual, predicted) {
	mae <- sum(abs(actual - predicted)) / length(actual)
	mae
}

# MAE Metric for XGBoost
getMAE <- function(preds, dtrain) {
  labels 	<- xgboost::getinfo(dtrain, "label")
  actval 	<- as.numeric(labels)
  predval 	<- as.numeric(preds)
  error 	<- calculateMAE(actval, predval)
  
  return(list(metric = "MAE", value = error))
}
 
# shift applying to the dependent variable ('loss')
shft <- 200
 
# Prepare list of categorical variables that are significant for prediction.
categoryList <- c(	"cat1","cat2","cat3","cat4","cat5",
					"cat6","cat7","cat9","cat10","cat11",
					"cat12","cat13","cat14","cat16","cat23",
					"cat24","cat25","cat28","cat36","cat38",
					"cat40","cat50","cat57","cat72","cat73",
					"cat76","cat79","cat80","cat81","cat82",
					"cat87","cat89","cat90","cat103","cat111")
 
 # Create cartesian product of Category variables.
categoryList <- merge(data.frame(f1 = categoryList), data.frame(f2 = categoryList))
categoryList <- data.table(categoryList)

# Keep only unique combination of features.
categoryList <- categoryList[as.integer(str_extract_all(f1, "[0-9]+")) > as.integer(str_extract_all(f2, "[0-9]+"))]

# Assign them back as character features
categoryList[, f1 := as.character(f1)]
categoryList[, f2 := as.character(f2)]

# Set seed for reproducibility
set.seed(456)

# Load Data files
traindata <- fread("train.csv", showProgress = TRUE)
testdata  <- fread("test.csv", showProgress = TRUE)
 
# Introduce loss feature in Test dataset
testdata$loss <- 0

# Combine both Test and Train datasets
fulldata <- rbind(traindata, testdata)

# Create new categorical variables
for (f in 1:nrow(categoryList)) {
  f1 <- categoryList[f, f1]
  f2 <- categoryList[f, f2]
  vrb <- paste(f1, f2, sep = "_")
  fulldata[, eval(vrb) := paste0(fulldata[[f1]], fulldata[[f2]])] 
}
 
# Eliminate skewness of data
for (f in colnames(fulldata)[colnames(fulldata) %like% "^cont"]) {
  tst <- e1071::skewness(fulldata[, eval(as.name(f))])
  if (tst > .25) {
    if (is.na(fulldata[, BoxCoxTrans(eval(as.name(f)))$lambda]))
		next
		
    fulldata[, eval(f) := BoxCox(fulldata[[f]], BoxCoxTrans(fulldata[[f]])$lambda)]
  }
}
 
# Disttribute data normally using 'scale'
for (f in colnames(fulldata)[colnames(fulldata) %like% "^cont"]) {
  fulldata[, eval(f) := scale(eval(as.name(f)))]
}

# Split Train and Test datasets from combined data
train		<- fulldata[fulldata$loss > 0,  ]
test		<- fulldata[fulldata$loss == 0, ]

# Store train dataset's 'loss' value
trainLabel 	<- log(train[, "loss"] + shft)

# Remove unused objects to relinquish memory
train$loss 	<- NULL
train$id 	<- NULL
test$loss 	<- NULL
test$id 	<- NULL
fulldata 	<- NULL
categoryList<- NULL

# Convert Character strings into numerical variables using factors
feature.names <- colnames(train)[colnames(train) %like% "^cat"]
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels 		<- unique(c(train[[f]], test[[f]]))
    train[[f]] 	<- as.integer(factor(train[[f]], levels=levels))
    test[[f]] 	<- as.integer(factor(test[[f]],  levels=levels))
  }
}

# Define number of folds
no_folds <- 10

# Create number of folds
folds 	 <- createFolds(1:nrow(train), k = no_folds)
 
# Create constants
print.every	<- 100
early_stop	<- 40

# Create palce-holder list object
predictions <- list()

# Create Parameter list to be passed to XGBoost Algorithm
xgbParams <- list(booster 		= "gbtree"
                   , objective 	= "reg:linear"
                   , subsample 	= 0.7
                   , max_depth 	= 12
                   , eta 		= 0.028
				   , colsample_bytree = 0.7
                   , min_child_weight = 100)
 

# Create XGBoost compatible matrix for Test dataset
testMatrx <- xgb.DMatrix(data=as.matrix(test))

# Remove unused objects to relinquish memory
test <- NULL

# Function to accept XGBoost parameters and predict loss values
xgbModel <- function (trainset, trainsetLabl, evalset, evalsetLabl, iter = 1) {
  
  print(paste0("Trainset:", nrow(trainset), ". TrainsetLabl: ", length(trainsetLabl)))
  print(paste0("Evalset: ", nrow(evalset), ". EvalsetLabl: ", length(evalsetLabl)))

  # Create XGBoost compatible matrix for Train dataset
  trainMatrx <- xgb.DMatrix(data=as.matrix(trainset), label=trainsetLabl)
  
  # Create XGBoost compatible matrix for Validation datasset
  evalMatrx  <- xgb.DMatrix(data=as.matrix(evalset), label=evalsetLabl)

  # Train XGBoost Model
  xgb <- xgb.train(params 		= xgbParams
                   , data 		= trainMatrx
                   , nrounds 	= 5000
                   , verbose 	= 1
                   , feval 		= getMAE
                   , watchlist 	= list(eval = evalMatrx, train = trainMatrx)
                   , maximize 	= FALSE
				   , early.stop.round = early_stop
				   , print.every.n = print.every
                   )
  
  # Predict value for Evaluation dataset
  evalsetPred	<- exp(predict(xgb, evalMatrx)) - shft
  evalsetLabl 	<- exp(evalsetLabl) - shft
  
  print(paste("Iteration ", i, " done. Score: ", calculateMAE(evalsetLabl, evalsetPred)))
  
  pred <-  exp(predict(xgb, testMatrx)) - shft
  pred
}
 
# Create 10 fold model and predict "loss" values 
for (i in 1:no_folds) {
  predictions[[i]] <- xgbModel(trainset    	= as.data.frame(train[-folds[i][[1]], ])
							, trainsetLabl 	= trainLabel[-folds[i][[1]],]$loss
							, evalset  	 	= as.data.frame(train[folds[i][[1]], ])
							, evalsetLabl  	= trainLabel[folds[i][[1]],]$loss
							, iter 		= i
						)
}
 
# Convert Prediction List to dataframe
predsDF <- as.data.frame(predictions)

# Create Submission file for average of all the models
submission 		<- read.csv("sample_submission.csv", colClasses = c("integer", "numeric"))
submission$loss <- apply(predsDF, 1, function(x) { mean(x, na.rm=TRUE) })
write.csv(submission, "Submission-6-XGB-Feature_10Fold.csv", row.names = FALSE, quote=FALSE)
