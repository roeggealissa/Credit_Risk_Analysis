# Credit_Risk_Analysis

# Loan Prediction Risk Analysis

### Introduction

Fast Lending, a peer to peer lending company, wants to use machine learning to predict credit risk. Management at this company believe this will provide a quicker and more efficent loan process and reduce the risk of defaulted loans by identifying risky customers. Jill, the lead Data Scientist at Fast Lending, has provided us with the necessary data to run several machine learning algorithms to determine if any are effective at predicting credit risk.

### Results

We ran six machine learning algorithms on the loan data provided by Fast Lending. The first three are using RandomOverSampler, SMOTE, and ClusterCentroids with the LogisticRegression classifier. The fourth method uses SMOTEENN which combines under and over sampling with the LogisticRegression classifier. The final two use  the BalancedRandomForestClassifier and the EasyEnsembleClassifier.

##### Over Sampling
![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/632602a47cffe945652fd1eeb6f7edd69a6ce380/Oversample_Balance.png)
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/632602a47cffe945652fd1eeb6f7edd69a6ce380/Oversample_Confusion.png)
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/632602a47cffe945652fd1eeb6f7edd69a6ce380/Oversample_Classification.png)


### Issues

sklearn has a known issue where many of the larger machine learning algorithms will kill the kernel if too much memory is alloted to the process. The ClusterCentroids portion of the code had to be run in Google Colab to ensure adaquate disk space and RAM for the process. No other algorithm had this issue.
