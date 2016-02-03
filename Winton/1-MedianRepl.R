#################################
#                               #
# Winton Stock Market Challenge #
#                               #
#################################

# ASSUMPTION: The required data files are downloaded from competition site and made available locally.
# COMPETITION SITE URL: https://www.kaggle.com/c/the-winton-stock-market-challenge

# Perform house-keeping
rm(list=ls())
gc()

# Set working directory
setwd("C:/courses/kaggle/Winton")

# Read train and test data sets from input CSV File
trainall <- read.csv("train.csv", stringsAsFactors=FALSE)  # 40000  x 211
testall <- read.csv("test_2.csv", stringsAsFactors=FALSE)  # 120000 x 147

# Read Submission CSV file to impute result
# NOTE: The following three lines of code took very long time to impute data and save to CSV File.
# So this has been commented out and the results are exported to a separate TEXT file
# This text file will be read by a Java program and written to a CSV file for submission.

### submission <- read.csv("sample_submission_2.csv", stringsAsFactors=FALSE)  # 120000 x 147
### sample_submission$Predicted <- as.numeric(replicate(120000, apply(trainall[, 147:208], 
###					FUN=median, MARGIN = 2)))

# Create a data frame with the values to be imputed and prefix them based on  submission file requirement
df <- as.data.frame(paste0("_", seq(1:61), ",", apply(trainall[, 147:208], FUN = median, MARGIN = 2)))

# Write the data frame to an intermediary Java file 
write.table(df, "result.txt", row.names=F, quote = F)
