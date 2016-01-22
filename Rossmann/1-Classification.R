##############################
#                            #
# ROSSMANN - SALES PREICTION #
#                            #
##############################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.

# Perform house-keeping
rm(list=ls())
gc()
setwd("C:/home/kaggle/Prudential")
library(caret)
library(rpart)
set.seed(23456)

# Load Data files
trainall <- read.csv("train.csv", stringsAsFactors = FALSE) # 59381
testall <- read.csv("test.csv", stringsAsFactors = FALSE)   # 19765

# Have this handy. Will be useful later
trainallRowNum <- nrow(trainall) 

# Introduce Response field to combine Train and Test
testall$Response <- 0 

# Combine Train and Test data sets into one
fulldata <- rbind(trainall, testall)

### Data Cleanup Activities ###

# Convert String value to number 
fulldata$Product_Info_2 <- as.number(as.factor(fulldata$Product_Info_2)) - 1

# Replace NAs with 0
fulldata[is.na(fulldata)] <- 0

# Split the combined data back to its original ones
trainall <- fulldata[1:trainallRowNum, ]
testall <-  fulldata[-(1:trainallRowNum), ]

# Remove the Variable that got introduced before combining.
testall$Response <- NULL

# Clean-up unused dataframe
rm(fulldata)

# Create Data partition on Train data to validate the model (70:30)
trainIndex <- createDataPartition(trainall$Id, p = 0.7, list = FALSE)
train <- trainall[ trainIndex,]  # 41569
test  <- trainall[-trainIndex,]  # 17812

# Store Test response for verification and remove it from data frame
testresponse <- test$Response

# Remove features from data frame
train$Id <- test$Id <- NULL
test$Response <- NULL

### Develop Model ###

# Develop Recursive-Partition (rpart) Classification Tree Model
classTreeModel <- train(Response ~ ., data = train, method = "rpart")

# Predict response for aportioned train data
trainresponse <- predict(classTreeModel, newdata = test)

# Validate prediction result with that of aportioned train data result 
confusionMatrix(round(classTreePrediction), trainresponse)

### Predict Response ###

# Predict response for aportioned train data
testallId <- testall$Id
testall$Id <- NULL
classTreePrediction <- predict(classTreeModel, newdata = testall)

# Create output file
result <- data.frame(Id = testallId, Response = classTreePrediction)
write.csv(result, 'rpart-submission-1.csv', row.names = F, quote = F)
