###############################
#                             #
# ROSSMANN - SALES PREDICTION #
#                             #
###############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/rossmann-store-sales

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/home/kaggle/Rossman")

# Load Packages required
library("lubridate")
library(xgboost)
library(methods)
library(reshape2)

# Set seed for reproducibility
set.seed(23456)

# Read Train, Test, and Store data sets from input CSV File
traindata <- read.csv("train.csv", header = T, stringsAsFactors = FALSE)
testdata  <- read.csv("test.csv",  header = T, stringsAsFactors = FALSE)
storedata <- read.csv("store.csv", header = T, stringsAsFactors = FALSE)


### Exploratory Data Analysis and Data Cleansing
# Merge Store Data with Train and Test sets
merged_train <- merge(train, store)
merged_test  <- merge(test, store)

# Factorize StoreType
merged_train$StoreType <- as.numeric(as.factor(merged_train$StoreType))
merged_test$StoreType  <- as.numeric(as.factor(merged_test$StoreType))

# Factorize Assortment
merged_train$Assortment<- as.numeric(as.factor(merged_train$Assortment))
merged_test$Assortment <- as.numeric(as.factor(merged_test$Assortment))

# Factorize StateHoliday
merged_train$StateHoliday <- as.numeric(as.factor(merged_train$StateHoliday))
merged_test$StateHoliday  <- as.numeric(as.factor(merged_test$StateHoliday))

# Factorize PromoInterval
merged_train$PromoInterval <- as.numeric(as.factor(merged_train$PromoInterval))
merged_test$PromoInterval  <- as.numeric(as.factor(merged_test$PromoInterval))


# Impute 0 to missing [NA] fields
# CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2SinceWeek, Promo2SinceYear
merged_train[is.na(merged_train)] <- 0
merged_test[is.na(merged_test)]   <- 0

# Remove data where Sales = 0 from training set
merged_train <- merged_train[merged_train$Sales > 0,]

# Convert CHR Date field into actual Date
merged_train$Date <- ymd(merged_train$Date)
merged_test$Date  <- ymd(as.Date(merged_test$Date, format="%m/%d/%Y"))

# Extract Day, Month, and Year from Date object and then remove Data object
merged_train$Day   <- day(merged_train$Date)
merged_train$Month <- month(merged_train$Date)
merged_train$Year  <- year(merged_train$Date)
merged_train$Date  <- NULL

merged_test$Day   <- day(merged_test$Date)
merged_test$Month <- month(merged_test$Date)
merged_test$Year  <- year(merged_test$Date)
merged_test$Date  <- NULL

# Separate Sales figure from merge data and convert it to log scale to bring down its dimensionality.
merged_train$Sales <- log1p(merged_train$Sales)

# Remove unused and unimportant fields
merged_train$Customers <- NULL

# Create DMatrix object from train data set - Will be used for model generation. 
merged_train_matrix <- xgb.DMatrix(data = data.matrix(merged_train), label = Sales) 

# Create Data partition on Train data to monitor performance of the model
trainIndex <- createDataPartition(merged_train$Store, p=0.1, list=FALSE) 
dtest <- merged_train[trainIndex, ]

# Create DMatrix object from dtest data set - Will be used for watch list. 
merged_dtest_matrix <- xgb.DMatrix(data = data.matrix(dtest), label = Sales) 

# Create watchlist
watchlist<-list(train = merged_train_matrix, test = merged_dtest_matrix, verbose=1)

# Prepare parameter list - - Will be used for model generation. 
# Add more parameters for fine tuning or for early stopping. 
param <- list(	objective		= "reg:linear", 
                booster			= "gbtree",
                eta			= 0.025,
                print.every.n	        = 5,
                 early.stop.round	= 25,
                max_depth		= 6,
                subsample		= 0.7,
                colsample_bytree        = 0.7
)

# Initialize number of rounds and folds. Try with a bigger number initially, and perform cross-validation. 
# This is required to identify 'Global-minimum' instead of getting stuck with 'Local-minimum' 
cv.round <- 2000
cv.nfold <- 5 

### Model Generation 
# Perform Cross-validation using the above params and objects 
xgbcv <- xgb.cv(data = merged_train_matrix, label = Sales, 
			nrounds = cv.round, param = param, 
			nfold = cv.nfold, metrics=list("rmse","auc"))
			
# Plot to visualize how the cross-validation is performing 
# plot(xgbcv$test.mlogloss.mean, type='l') 

# Determine 'Global-minimum' number of rounds required for the model 
nround <- which(xgbcv$test.mlogloss.mean == min(xgbcv$test.mlogloss.mean) ) 

# Train a model using XGBoost Train and the above params / objects. Include advanced params
xgb  <- xgb.train(  data            = merged_train_matrix,
                    params          = param,
                    nrounds         = nround,
                    verbose         = 1,
                    watchlist       = watchlist,
                    maximize        = FALSE,
                    nfold 	    = cv.nfold,  ##
                    feval	    = RMPSE)
					
## Prepare Test data
# Store ID Column in separate variable and remove the same from data Set
Id <- test$Id
merged_test$Id <- NULL

## Prediction
# Conver Data.frame to Matrix
merged_test_matrix <- as.matrix(merged_test)

# Predict Sales
ypred <- predict(xgb, merged_test_matrix)

# Computed sales value will be in logarithmic format. Convert it to actual value
ypred <- expm1(ypred)

### Output
# Create a Data frame with ID and Prediction
salesdata <- data.frame(Id=Id, Sales=ypred)

# Write to Output file
write.csv(salesdata, row.names = F, quote = F, file = "3-XGBoost.csv")
