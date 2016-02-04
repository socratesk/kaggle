## Welcome to my Kaggle GitHub Page
I created my Kaggle account in Apr 2015 but really started participating in competitions from November 2015. My first competition was predicting 'Rossmann Store Sales' that ran through mid-December 2015. I tried basic Classification algorithm for my first submission to that competition to verify my position in Public Leader Board. Then slowly accelerated to other types of models.

## Competitions Participated
* Rossmann Store Sales - Sales predicting competition. Competition status: **Closed.**
* Walmart Recruiting: Trip Type Classification - Classification of Customer Trip to Walmart stores. Competition status: **Closed.**
* The Winton Stock Market Challenge - Predicting intra and end of day stock returns future. Competition status: On-going.
* Prudential Life Insurance Assessment - Customer Risk classification on Life Insurance application. Competition status: On-going.
* Telstra Network Disruptions -  Predict the severity of service disruptions on the network. Competition status: On-going.

## Models used
Before participating in the above competitions, I analyzed the previous Kaggle competitions and user scripts. Predominantly, 'XGBoost' was used for single model submissions in both competitions. So I tried 'XGBoost' to start with and ensembles of XGBoost, Random Forest, Neural Network, Classification, and GBM.

**Rossmann Store Sales**
  1. Linear Model - [Link](https://github.com/socratesk/kaggle/blob/master/Rossmann/1-LinearModel.R)
  2. XGBoost with Feature Engineering - [Link](https://github.com/socratesk/kaggle/blob/master/Rossmann/2-XGBoost-FeatureEngg.R)
  3. XGBoost with Feature Engineering and Parameter tuning - [Link](https://github.com/socratesk/kaggle/blob/master/Rossmann/3-XGBoost-FeatureEngg-ParamTune.R)

**Prudential Life Insurance**
  1. Recursive Partition Classification Tree - [Link](https://github.com/socratesk/kaggle/blob/master/Prudential/1-Classification.R)
  2. Random Forest - [Link](https://github.com/socratesk/kaggle/blob/master/Prudential/2-RandomForest.R)

**Walmart Recruiting**
  1. XGBoost with Feature Engineering - [Link](https://github.com/socratesk/kaggle/blob/master/Walmart-1/1-XGBoost-FeatureEngg.R)
  2. Random Forest with Feature Engineering - [Link](https://github.com/socratesk/kaggle/blob/master/Walmart-1/2-RandomForest-FeatureEngg.R)

**The Winton Stock Market Challenge**

Working in this competition was bit more challenging than others due to the file size and volume of data.<br>
* Train data set: 40000 rows x 211 features. File size: 173 MB<br>
* Test data set: 120000 rows x 147 features. File size: 282 MB


With initial model, preparation of output file was taking hours to process. So had to findout alternate ways to process them faster. To do that task, ended-up developing Java code 

  1. Median replacement - [R Code](https://github.com/socratesk/kaggle/blob/master/Winton/1-MedianRepl.R); [Java Code](https://github.com/socratesk/kaggle/blob/master/Winton/1_Output.java)
  2. Median replacement 3% Adjusted - [R Code](https://github.com/socratesk/kaggle/blob/master/Winton/2-Adjusted_3perc_MedianRepl.R)
  3. Moving average on Test data - [R Code](https://github.com/socratesk/kaggle/blob/master/Winton/3-MovingAverage.R); [Java Code](https://github.com/socratesk/kaggle/blob/master/Winton/3_Csvfile.java)

## Focus
Kaggle's recruiting competition prohibits participants to post their code in the forum and form a group or team. This helps each individual to exhibit their own idea and expertise in Data Science area - not hijacked by others idea. This really helps me evaluate where I stand, as an individual, when compared to other participants across the World. 
