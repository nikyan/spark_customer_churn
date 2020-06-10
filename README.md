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




