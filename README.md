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

##### SMOTE

![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTE_Balance.png)
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTE_Confusion.png)
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTE_Classification.png)

##### Undersampling

![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/CC_balance.png)
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/CC_confusion.png)
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/CC_Classification.png)

##### SMOTEENN

![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTEENN_balance.png)
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTEENN_Confusion.png)
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/2666e62ac16aa367f69cfd329d4588efa9879ccf/SMOTEENN_Classification.png)

##### Random Forest

![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/RF_balance.png)
<br/><br/>
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/RF_Confusion.png)
<br/><br/>
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/RF_Classification.png)

##### Easy Ensemble

![Balance](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/EEC_balance.png)
![Confusion](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/EEC_Confusion.png)
![Classification](https://github.com/roeggealissa/Credit_Risk_Analysis/blob/a51f263b1d0e70a160bb8809b1459763582e0953/EEC_Classification.png)

### Conclusions

All models have a low precision for the high risk credit for a loan. The undersampling model has the worst recall with an avg/total at 0.40 with the high risk recall contributing the most to it also at 0.40. The Easy Ensemble method had the highest recall at 0.94, with a recall of 0.94 and 0.91 for the high risk and low risk respectively. The Easy Ensemble method also has the highest F1 score at 0.97. The Easy Ensemble method has the highest balance accuracy score at 0.925. SMOTEENN, SMOTE, and Oversampling all have similar results, but fair poorer than the Easy Ensemble and Random Forest Method. The Random Forest Method has a high avg/total recall score at 0.91, but the recall for high risk is low at 0.67.

Overall the suggestion is to use the Easy Ensemble Classifier method due to it's overall superiority in every catagory.

### Issues

sklearn has a known issue where many of the larger machine learning algorithms will kill the kernel if too much memory is alloted to the process. The ClusterCentroids portion of the code had to be run in Google Colab to ensure adaquate disk space and RAM for the process. No other algorithm had this issue.
