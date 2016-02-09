# Perform house-keeping
print('House-keeping... ')
rm(list=ls())
gc()

# Load Packages required
print('Loading Libraries... ')
library(readr)
library(xgboost)

# Set seed for reproducibility
print('Setting Seeds... ')
set.seed(23456)


### Read Train and Test Data sets
print('Reading data sets... ')
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

# Introduce new column in Test data for Data Cleansing and Feature Engineering
test$QuoteConversion_Flag <- 8

# Store Quote number in a Separate variable. Will be used in Final Submission file
testQuoteNumber <- test$QuoteNumber

# Combine Train and Test data sets
fulldata <- rbind(train, test)

# Clean-up original data sets
rm(train)
rm(test)

### Perform Feature Engineering
print('Performing Feature Engineering... ')
# Convert Date String to Date Object
fulldata$Original_Quote_Date <- as.Date(fulldata$Original_Quote_Date)
fulldata$month <- as.integer(format(fulldata$Original_Quote_Date, "%m"))
fulldata$year  <- as.integer(format(fulldata$Original_Quote_Date, "%y"))
fulldata$day   <- weekdays(as.Date(fulldata$Original_Quote_Date))

# Remove data object as we have extracted more feature from it
fulldata$Original_Quote_Date <- NULL

# Perfrom Dimensional Reduction and Expansion
print('Performing Dimensional Reduction ... ')
fulldata$Field9  		<- fulldata$Field9 * 10000
fulldata$Field10 		<- as.numeric(gsub(",", "", fulldata$Field10)) / 100
fulldata$SalesField8 	<- log(fulldata$SalesField8)

# Remove data that has only one type of values
fulldata$PropertyField6 	<- NULL
fulldata$GeographicField10A	<- NULL
fulldata$QuoteNumber 		<- NULL

# Imputes NAs with -1 
fulldata[is.na(fulldata)]	<- -1  ### To be kept as 0 or as -1?

# Convert Character features into Numerical
print('Converting Character features into Numerical... ')
feature.names <- names(fulldata)
for (f in feature.names) {
  if (class(fulldata[[f]]) == "character") {
    levels <- fulldata[[f]]
    fulldata[[f]] <- as.integer(factor(fulldata[[f]], levels = levels))
  }
}

# Split data to its Original sets
print('Splitting Train and test data sets ... ')
train <- fulldata[fulldata$QuoteConversion_Flag != 8, ]
test  <- fulldata[fulldata$QuoteConversion_Flag == 8, ]

# Remove newly introduced flag
test$QuoteConversion_Flag <- NULL

# Some cleanup before generating models
rm(fulldata)
