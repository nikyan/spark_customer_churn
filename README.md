# Big data analytics: Predict Customer Churn

Predict customer churn from customer's log data using Apache Spark on a Spark cluster. In this project I am using Sparkify data from Udacity which is a dummy customer data mimicking event log of an application like spotify. 

## Motivation for the Project:
 - Use Apache Spark on event log data to extract features for machine learning
 - Train machine learning algorithm to predict customer churn
 - Perform model evaluation using hyper parameter tuning
 - Apply process mining algorithm on event log data to visualize the page events
 
 ## Files in this repository:
  - sparkify_EDA.ipynb: Jupyter Notebook containing explorative data analysis using mini sparkify data
  - sparkify_pipeline.ipnyb: Jupyter Notebook containing ETL script and ML script to extract features, train and evaluate models using mini sparkify data
  - sparkify_pipeline_gcp-cluster.ipnyb: Jupyter Notebook containing results of ETL and ML script run on full dataset using GCP's Dataproc cluster
  - sparkify_process_mining.ipnyb: Jupterh Notebook containing implmentation of various process mining algorithms using PMP4y library
  
  ## Prerequisites
  
  Libraries used:
  - Pandas/Numpy
  - Matplotlib/Seaborn
  - Datetime
  - PySpark
  - PMP4y
  
  Google Colab:
  Setup Google Colab for using Spark on single machine using instructions here:
  https://colab.research.google.com/github/asifahmed90/pyspark-ML-in-Colab/blob/master/PySpark_Regression_Analysis.ipynb
  
  ## Exploratory Data Analysis (EDA)
  
In EDA using mini dataset, the main goal is to get a better understanding of the data. Identify steps that need to be performed in order to make data ready for machine learning.

mini sparkify data: This is a small dataset consisting of approximately 2 months of data.

1. Idenfity primary key and look for nulls.
'userId' is the primary key and churn needs to be predicted for each user.

![GitHub Logo](/images/null_check.png)

2. Identify churned user and evaluate various features on active or churned user
 
![GitHub Logo](/images/churn_rate.png)

In EDA, I evaluate following features:
- gender
- level
- location: this field requires clean up to make it more meaningful
- userAgent: this field requires clean up to make it more meaningful
- status
- days since registration
- total sessions per user
- artists/songs
- page views (monthly and daily average views)

## Feature Extraction

Extract features from the dataset that will be used for training classification models

 
 - |-- gender: string (nullable = true)
 - |-- label: integer (nullable = true)
 - |-- level: string (nullable = true)
 - |-- state: string (nullable = true)
 - |-- device: string (nullable = true)
 - |-- days_since_reg: integer (nullable = true)
 - |-- avg_mon_session_count: double (nullable = true)
 - |-- avg_mon_sess_duration: double (nullable = true)
 - |-- avg_daily_About: double (nullable = false)
 - |-- avg_daily_Add_Friend: double (nullable = false)
 - |-- avg_daily_Add_to_Playlist: double (nullable = false)
 - |-- avg_daily_Downgrade: double (nullable = false)
 - |-- avg_daily_Error: double (nullable = false)
 - |-- avg_daily_Help: double (nullable = false)
 - |-- avg_daily_Home: double (nullable = false)
 - |-- avg_daily_Logout: double (nullable = false)
 - |-- avg_daily_NextSong: double (nullable = false)
 - |-- avg_daily_Roll_Advert: double (nullable = false)
 - |-- avg_daily_Save_Settings: double (nullable = false)
 - |-- avg_daily_Settings: double (nullable = false)
 - |-- avg_daily_Submit_Downgrade: double (nullable = false)
 - |-- avg_daily_Submit_Upgrade: double (nullable = false)
 - |-- avg_daily_Thumbs_Down: double (nullable = false)
 - |-- avg_daily_Thumbs_Up: double (nullable = false)
 - |-- avg_daily_Upgrade: double (nullable = false)
 - |-- avg_mon_About: double (nullable = false)
 - |-- avg_mon_Add_Friend: double (nullable = false)
 - |-- avg_mon_Add_to_Playlist: double (nullable = false)
 - |-- avg_mon_Downgrade: double (nullable = false)
 - |-- avg_mon_Error: double (nullable = false)
 - |-- avg_mon_Help: double (nullable = false)
 - |-- avg_mon_Home: double (nullable = false)
 - |-- avg_mon_Logout: double (nullable = false)
 - |-- avg_mon_NextSong: double (nullable = false)
 - |-- avg_mon_Roll_Advert: double (nullable = false)
 - |-- avg_mon_Save_Settings: double (nullable = false)
 - |-- avg_mon_Settings: double (nullable = false)
 - |-- avg_mon_Submit_Downgrade: double (nullable = false)
 - |-- avg_mon_Submit_Upgrade: double (nullable = false)
 - |-- avg_mon_Thumbs_Down: double (nullable = false)
 - |-- avg_mon_Thumbs_Up: double (nullable = false)
 - |-- avg_mon_Upgrade: double (nullable = false)

'label' is what we are trying to predict. '0' represents active users and '1' represents churned users.

## Correlation analysis
Since there are so many features that I extract it's best to look at correlation between various features. Highly correlated features will not be useful for model training.

![GitHub Logo](/images/correlations.png)

Another way to understand highly correlated features is to visualize log data especially page views using process mining algorithms.

![GitHub Logo](/images/directly_follows_graph.png)

The Directly Follows graph is a great way to analyze process and get an idea of sequence of page events. It's also good way to see highly correlated features. For e.g. you can see 'Cancel' is always followed by 'Cancellation Confirmation' page.


## PCA
Perform PCA to evaluate feature importance. While correlation analysis provides insights into highly correlated features, it's easier to use PCA for feature extraction.

![GitHub Logo](/images/scree_plot.png)

10 features covers 85% of variability in the data. I will train the models using top 10 PCA features.

## Train models using baseline parameters

Use F1 score for model evaluation. Since this is an imbalanced dataset, f1 is the better metric compared with accuracy.
It's important to check whether the models are predicting both majority and minority class. Since it's imbalanced data the model may only predict majority class and yet give a high accuracy rate.

## Train with full dataset on GCP Dataproc cluster

Full dataset provides better result from classification models. I used GCP's Dataproc to provision 3 node cluster and run 'sparkify_pipeline_gcp-cluster.ipnyb' on full dataset. 

## Hyper Parameter Tuning

Performed parameter tuning on Logistic Regression, Random Forest Classifier, GBT Classifier and Decision Tree Classifier. Random Forest performs the best and provides highest gain in f1 metric post hyper parameter tuning.

*Baseline model:*
- f1 score for Random Forest Classifier: train dataset 0.724383305188, test dataset 0.725493733818

*Post hyper parameter tuning:*
- f1 train score: 0.912099551257
- f1 test score: 0.83646658194
- Best parameter for max depth: 15
- Best parameter for number of tress: 200
- Best parameter for number of bins: 50




