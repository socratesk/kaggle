

For Homesite Quote Conversion Competition, I tried at least 3 modeling techniques individually and collectively (ensemble). Tried some feature engineering as well in all the above techniques.
Following are the modeling techniques I tried:
a. GBM
b. MLR
c. XGBoost

For ensemble, I split the Train data into two parts. Applied all the above techniques in the first part of Train data set (train50) and predicted the outcome in the second part of Train data set (test50). Again used those predicted values as features in the second part. I applied the same technique in actual Test data set as well.

Then used XGBoost technique in the second part of train data set (containing new predicted values from previous models) and arrived at the final model. Applied this final model on Test data set to arrive at final prediction.
My approach in the form of visual representation …

