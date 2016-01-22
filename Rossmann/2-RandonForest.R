###############################
#                             #
# ROSSMANN - SALES PREDICTION #
#                             #
###############################

# Perform house-keeping
rm(list=ls())
gc()
setwd("C:/home/kaggle/Prudential")
library(caret)
library(rpart)
set.seed(23456)

# Load Data files
trainall <- read.csv("train.csv", stringsAsFactors=FALSE) # 59381
testall <- read.csv("test.csv", stringsAsFactors=FALSE)   # 19765

# Have this handy. WIll be usefule later
trainallRowNum <- nrow(trainall) 

# Introduce Response field to combine Train and Test
testall$Response <- 0 

# Combine Train and Test data sets into one
fulldata <- rbind(trainall, testall)

### Data Cleanup activities ###

# Convert String value to number 
fulldata$Product_Info_2 <- as.number(as.factor(fulldata$Product_Info_2)) - 1

# Replace NAs with 0
fulldata[is.na(fulldata)] <- 0

# Split the combined data back to its original ones
trainall <- fulldata[1:trainallRowNum, ]
testall <-  fulldata[-(1:trainallRowNum), ]

# Remove the Variable that got intriduced before combining.
testall$Response <- NULL

# Clean-up unused dataframe
rm(fulldata)

# Create Data partition on Train data to validate the model (70:30)
trainIndex <- createDataPartition(trainall$Id, p=0.7, list=FALSE)
train <- trainall[ trainIndex,]  # 41569
test  <- trainall[-trainIndex,]  # 17812

# Store Test response for verification and remove it from data frame
testresponse <- test$Response

# Remove features from data frame
train$Id <- test$Id <- NULL
test$Response <- NULL

# Develope Recursive-Partition (rpart) Classification Tree Model
rfmodel <- train(Response ~ ., data = train[, -ncol(train)], method = "rf")

# Predict response for aportioned train data
testpred <- predict(rfmodel, newdata = test)

# Validate prediction result with that of aportioned train data result 
confusionMatrix(round(testpred), testresponse)

# Predict response for aportioned train data
testallId <- testall$Id
testall$Id <- NULL
testallpred <- predict(rfmodel, newdata = testall)

# Create output file
result <- data.frame(Id = testallId, Response = testallpred)
write.csv(result, 'rf-submission-2.csv', row.names=F, quote = F)
