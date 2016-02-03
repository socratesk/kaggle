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

# Read test data set from input CSV File
testall <- read.csv("test_2.csv", stringsAsFactors=FALSE)  # 120000 x 147

# Create an empty data frame that will be used for output
df <- data.frame(Id=c(), Predicted=c())

# Initialise starting postion and interval for moving average calculation
startpos <- 28
interval <- 59

# Initialize incrementor
i <- 1

for (j in 1:60) {

	# Compute window frame
	startpos <- startpos + i
	endpos <- startpos + interval
	cat('\n', j, ' Start Pos: ', startpos, ' End Pos: ', endpos)

	# Compute median value for "moving Median"
	# med <- apply(testall[1:nrow(testall), startpos:endpos], MARGIN = 1,
	#		 function(x) median(na.omit(x)))

	# Compute average value
	avg <- apply(testall[1:nrow(testall), startpos:
	
	# Prepare ID value for sorting
	numchar <- ifelse (j < 10, paste('0', j, sep=''), j)
	ids <- paste(c(1:length(avg)), '_', numchar, sep="")

	# insert ID and calculated average in data frame
	df <- rbind(df, data.frame(Id = ids, Predicted = med))
}

# Compute fluctuaion for 2 hours (Given value is for every minute)
retplusone <- apply(testall[1:nrow(testall), 29:147], MARGIN = 1,
			  function(x) sum(na.omit(x)))

# Compute end of day fluctuation (for 8 hours)
retplusone <- retplusone * 4

# Prepare this return to append to existing data frame
retplusoneids <- paste(c(1:length(retplusone)), '_', 61, sep="")

# Append the value in existing data frame
df <- rbind(df, data.frame(Id = retplusoneids, Predicted = retplusone))

# Prep work for daily average computation
retdf <- data.frame(testall[1:nrow(testall), 27:28])
retdf <- cbind(retdf, retplusone=retplusone)

# Conmpute daily average
retplustwo <- apply(retdf, MARGIN = 1, function(x) median(na.omit(x))) ### Median
retplustwoids <- paste(c(1:length(retplustwo)), '_', 62, sep="")

# Append daily average in existing data frame
df <- rbind(df, data.frame(Id = retplustwoids, Predicted = retplustwo))

# Write the final output to a CSV file 
# This will be read by another Java file for sorting and final oiutput
df$Predicted <- format(df$Predicted, scientific=FALSE)
write.table(df, 'result.csv',  quote = FALSE, row.names=F, sep=',', col.names=FALSE)
