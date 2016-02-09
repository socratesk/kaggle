###############################
#                             #
#  HOMESITE QUOTE CONVERSION  #
#                             #
###############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/homesite-quote-conversion

# Set working directory
setwd("C:/courses/kaggle/Homesite")

# Loads Train and Test Data sets and does some Feature Engineering 
source("_data.R")

# Load specific libraries
library(caret)
library(gbm)
library(mlr)

# Call-out response variable
Response <- "QuoteConversion_Flag"

# Split Train dataset into train50 and test50 [50-50]
trainindex50 <- createDataPartition(train$SalesField8, p=0.5, list=FALSE)
train50 <- train[ trainindex50, ]
test50  <- train[-trainindex50, ]

# Store Target feature in a separate variable for Cross-Validation
test50QuoteConversion_Flag <- test50$QuoteConversion_Flag
test50$QuoteConversion_Flag <- NULL
feature.names <- names(test50)

### Generate Ensemble Models
## 1. Develop GBM Model 
gbm_model <- gbm(QuoteConversion_Flag ~ ., 
				data	= train50, 
				dist	= "gaussian", 
				n.trees = 1000,
				shrinkage = 1, 
				train.fraction = 0.5)


## 2. Develop MLR Model
# Create mlr task and convert factors to dummy features
trainTask = makeRegrTask(data = train50, target = "QuoteConversion_Flag")
trainTask = createDummyFeatures(trainTask)

# Create MLR learner
lrn = makeLearner("regr.xgboost")
	lrn$par.vals = list(
	nthread         = 5,
	nrounds         = 1800,
	print.every.n	= 2,
	objective       = "count:poisson"
)

# Impute missing values by their median
lrn = makeImputeWrapper(lrn, classes = list(numeric=imputeMedian(), integer=imputeMedian()))

# Train the model on train50 data
mlr_model = train(lrn, trainTask)

## 3. Develop XGB Model 
# Create some random index for XGBoost's Watch-list 
eval <- train50[sample(nrow(train50), 5000), ]

# Create DMatrix for Train data
train50matrix <- xgb.DMatrix(data = data.matrix(train50[, names(train50) != Response]),
			label = train50$QuoteConversion_Flag)

# Create DMatrix for Eval data
evalmatrix  <- xgb.DMatrix(data = data.matrix(eval[, names(eval)  != Response]), 
			label = eval$QuoteConversion_Flag)

# Initial XGB's Rounds and folds.
cv.round <- 2000
cv.nfold <- 5

# Create Watchlist object
watchlist <- list(val = evalmatrix, train = train50matrix)

# Create Param List for XGB
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.023,
                watchlist           = watchlist,
                max_depth           = 6,
                subsample           = 0.83,
                colsample_bytree    = 0.77 
)

# Perform Cross-validation using the above params and objects
 xgbcv <- xgb.cv(data = train50matrix, #label = QuoteConversion_Flag,
				nrounds = cv.round, param = param,
				nfold=cv.nfold)

# Choose mean AUC for no of rounds
nround <- which(xgbcv$test.auc.mean == max(xgbcv$test.auc.mean) )

# Train XGB Model
xgb_model <- xgb.train(params           = param, 
                    data                = train50matrix, 
                    nrounds             = nround,  #1986
                    verbose             = 1,  #1
                    early.stop.round    = 40,
                    watchlist           = watchlist,
                    maximize            = FALSE
)


### Perform Predictions using three models on remaining 50% of Test set
test50$GBM_Pred <- predict(gbm_model, test50[, feature.names])
test50$XGB_Pred <- predict(xgb_model, data.matrix(test50[, feature.names]))

# Prediction of values in MLR requires creation of Task with Response variable
test50$QuoteConversion_Flag <- 0  # Replace the original values into 0, forcibly
feature.names.qcf <- c(feature.names, Response)

# Create Test50Task for MLR prediction 
test50Task  = makeRegrTask(data = test50[, feature.names.qcf], target = "QuoteConversion_Flag")
test50Task  = createDummyFeatures(test50Task)
test50$MLR_Pred = (predict(mlr_model, test50Task))$data$response

# Now, attach original value of Response variable in test50. Will be used for final model generation
test50$QuoteConversion_Flag <- test50QuoteConversion_Flag


### Perform Predictions using three models on original Test set
test$GBM_Pred <- predict(gbm_model, test[, feature.names])
test$XGB_Pred <- predict(xgb_model, data.matrix(test[, feature.names]))

# Prediction of values in MLR requires creation of Task with Response variable. So impute dummy value
test$QuoteConversion_Flag <- 0

# Create Test50Task for MLR prediction 
testTask  = makeRegrTask(data = test[, feature.names.qcf], target = "QuoteConversion_Flag")
testTask  = createDummyFeatures(testTask)
test$MLR_Pred = (predict(mlr_model, testTask))$data$response

# Remove Response value from Test Data
test$QuoteConversion_Flag <- NULL


### Creation of Final Model with new predictions added in Test50 data set
## Create some random index for XGBoost's Watch-list validation
evaltest50 <- test50[sample(nrow(test50), 5000), ]

# Create DMatrix for Train data
test50matrix <- xgb.DMatrix(data = data.matrix(test50[, names(test50) != Response]),
			label = test50$QuoteConversion_Flag)

# Create DMatrix for Eval data
evaltest50matrix  <- xgb.DMatrix(data = data.matrix(evaltest50[, names(evaltest50)  != Response]), 
			label = evaltest50$QuoteConversion_Flag)
			
# Initial XGB's Rounds and folds.
cv.round <- 2300
cv.nfold <- 5

# Create Watchlist object
watchlist <- list(val = evalmatrix, train = test50matrix)

# Create Param List for XGB
param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = c("auc","mlogloss")
                eta                 = 0.023,
                watchlist           = watchlist,
                max_depth           = 6,
                subsample           = 0.83,
                colsample_bytree    = 0.77 
)

# Perform Cross-validation using the above params and objects
xgbcv <- xgb.cv(data = test50matrix, #label = QuoteConversion_Flag,
				nrounds = cv.round, param = param,
				nfold=cv.nfold)

# Choose mean AUC for no of rounds
nround <- which(xgbcv$test.auc.mean == max(xgbcv$test.auc.mean) )

# Train XGB Model
xgb_final_model <- xgb.train(params     = param, 
                    data                = test50matrix, 
                    nrounds             = nround, # 500
                    verbose             = 1,
                    early.stop.round    = 40,
                    watchlist           = watchlist,
                    maximize            = FALSE
)

## Perform Final Predictions
xgb_final_model_Pred <- predict(xgb_final_model, data.matrix(test[, names(test)  != Response]))

## Create submission file
submission = data.frame(QuoteNumber = testQuoteNumber)
submission$QuoteConversion_Flag <- format(xgb_final_model_Pred, scientific = FALSE)
write.csv(submission, "Submission-Ensemble-GBM-XGB-MLR-XGB.csv", quote = FALSE, row.names = FALSE)
