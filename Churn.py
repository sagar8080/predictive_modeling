# Sagar's Project on Churn prediction using Predictive ModellingTechniques

# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the data from csv file

data = pd.read_csv('churn-data.csv')
data.head()
data.info()

# Data PreProcessing

data['TotalCharges']=pd.to_numeric(data['TotalCharges'], errors = 'coerce')
data.loc [data['TotalCharges'].isna()== True]
data.loc [data['TotalCharges'].isna()== True] = 0
data['OnlineBackup'].unique()

# convert into categorical value for Algo to Process

data['Partner'].replace(['Yes','No'],[1,0],inplace=True)
data ['gender'].replace(['Male','Female'], [1,0],inplace = True)
data['Dependents'].replace(['Yes','No'],[1,0],inplace=True)
data['PhoneService'].replace(['Yes','No'],[1,0],inplace=True)
data['MultipleLines'].replace(['No phone service','No', 'Yes'],[0,0,1],inplace=True)
data['InternetService'].replace(['No','DSL','Fiber optic'],[0,1,2],inplace=True)
data['OnlineSecurity'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['OnlineBackup'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['DeviceProtection'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['TechSupport'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingTV'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['StreamingMovies'].replace(['No','Yes','No internet service'],[0,1,0],inplace=True)
data['Contract'].replace(['Month-to-month', 'One year', 'Two year'],[0,1,2],inplace=True)
data['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace=True)
data['PaymentMethod'].replace(['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'],[0,1,2,3],inplace=True)
data['Churn'].replace(['Yes','No'],[1,0],inplace=True)
 
data.pop('customerID')
data.info()

# Correlation between Customer data features and Customer Churn

corr = data.corr()
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values, annot = True, annot_kws = {'size' :12})
heatmap = plt.gcf()
heatmap.set_size_inches(10, 10)
plt.xticks(fontsize = 5)
plt.yticks(fontsize = 5)
plt.show()

# Multi Co-linearity is a problem for implementing Regression
# and perfect multicollinearity occurs when the correlation between two independent variables is equal to 1 or -1.

# PREDICTIVE MODELLING

from sklearn.model_selection import train_test_split
train , test = train_test_split (data, test_size = 0.26 )

train_y = train['Churn']
test_y = train['Churn']

train_x = train
train_x.pop('Churn')
test_x = test
test_x.pop('Churn')

# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

lr = LogisticRegression()
lr.fit(X = train_x ,y = train_y)
test_y_pred = lr.predict(test_x)
confusion_matrix = confusion_matrix(test_y,test_y_pred)
print("intercept =" + str(lr.intercept_))
print("Regression = " + str(lr.coef_))
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr.score(test_x, test_y)))
print (classification_report(test_y, test_y_pred))
    
confusion_matrix_df = pd.DataFrame(confusion_matrix, ('No churn', 'Churn'), ('No churn', 'Churn'))
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize = 14)
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
plt.show()


# Handle imbalanced datasets

data['Churn'].value_counts()
# Around 25 % of popn.
# Up sample the minority

from sklearn.utils import resample 
data_majority = data[data['Churn'] == 0]
data_minority = data[data['Churn'] == 1 ]
data_minority_upsampled = resample(data_minority, replace = True, n_samples = 5174 , random_state = 1)
data_upsampled = pd.concat([data_majority , data_minority_upsampled])
data_upsampled['Churn'].value_counts()

# now there is a 1:1 data; apply Logistic Regression to this again

train, test = train_test_split(data_upsampled, test_size = 0.25)

train_y_upsampled = train['Churn']
test_y_upsampled = test['Churn']
train_x_upsampled = train
train_x_upsampled.pop('Churn')
test_x_upsampled = test
test_x_upsampled.pop('Churn')
 
logisticRegr_balanced = LogisticRegression()
logisticRegr_balanced.fit(X=train_x_upsampled, y=train_y_upsampled)
 
test_y_pred_balanced = logisticRegr_balanced.predict(test_x_upsampled)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticRegr_balanced.score(test_x_upsampled, test_y_upsampled)))
print(classification_report(test_y_upsampled, test_y_pred_balanced))

# AUROC

from sklearn.metrics import roc_auc_score

# Get class probabilities for both models
lr.fit(test_x,test_y)
test_y_prob = lr.predict_proba(test_x)
test_y_prob_balanced = lr_balanced.predict_proba(test_x_upsampled)

# We only need the probabilities for the positive class
test_y_prob = [p[1] for p in test_y_prob]
test_y_prob_balanced = [p[1] for p in test_y_prob_balanced]

print('Unbalanced model AUROC: ' + str(roc_auc_score(test_y, test_y_prob)))
print('Balanced model AUROC: ' + str(roc_auc_score(test_y_upsampled, test_y_prob_balanced)))


from sklearn import tree
from sklearn import tree
import graphviz 
 
# Create each decision tree (pruned and unpruned)
decisionTree_unpruned = tree.DecisionTreeClassifier()
decisionTree = tree.DecisionTreeClassifier(max_depth = 4)
 
# Fit each tree to our training data
decisionTree_unpruned = decisionTree_unpruned.fit(X=train_x, y=train_y)
decisionTree = decisionTree.fit(X=train_x, y=train_y)
 
# Generate PDF visual of decision tree
churnTree = tree.export_graphviz(decisionTree, out_file=None, 
                         feature_names = list(train_x.columns.values),  
                         class_names = ['No churn', 'Churn'],
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(churnTree)
graph.render('decision_tree.gv', view=True)
test_y_pred_dt = decisionTree.predict(test_x)
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(decisionTree.score(test_x, test_y))

