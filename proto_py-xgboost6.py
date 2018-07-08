#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 13:30:58 2017

@author: mohit
"""

#proto first pyhton submission


#load libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.model_selection import *
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt



#load data
train = pd.read_csv("data/train.csv") #test data
test = pd.read_csv("data/test.csv") #train data
sample_submission = pd.read_csv("data/sample_submission.csv")


#train.iloc[train['ps_ind_02_cat'].idxmin()]

#filling missing values "-1 or -1.0" with NaN
train[train == -1] = np.nan
train[train == -1.0] = np.nan


test[test == -1] = np.nan
test[test == -1.0] = np.nan

#look for null NaN
train.isnull().sum()
test.isnull().sum()


#convert cat NA to most_frequesnt
from sklearn.preprocessing import Imputer

#NaN for flaot and integer

imputer_reg_03 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_reg_03.fit(train.iloc[:, 22:23])
train.iloc[:, 22:23] = imputer_reg_03.transform(train.iloc[:, 22:23])

imputer_car_11 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_11.fit(train.iloc[:, 34:35])
train.iloc[:, 34:35] = imputer_car_11.transform(train.iloc[:, 34:35])

imputer_car_12 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_12.fit(train.iloc[:, 35:36])
train.iloc[:, 35:36] = imputer_car_12.transform(train.iloc[:, 35:36])

imputer_car_14 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_14.fit(train.iloc[:, 37:38])
train.iloc[:, 37:38] = imputer_car_14.transform(train.iloc[:, 37:38])

#NaN for Cat. add NaN as extra LEVEL

cat_level_values = {'ps_ind_02_cat': 5,'ps_ind_04_cat': 2, 'ps_ind_05_cat': 7, 'ps_car_01_cat': 12, 'ps_car_02_cat': 2,
                    'ps_car_03_cat': 2, 'ps_car_04_cat' : 10, 'ps_car_05_cat' : 2, 'ps_car_06_cat': 18, 'ps_car_07_cat': 2,
                    'ps_car_08_cat': 2, 'ps_car_09_cat': 5, 'ps_car_10_cat': 3, 'ps_car_11_cat': 105}

train.fillna(value=cat_level_values, inplace=True)



train.isnull().sum()

#remove calculated variable

train = train.iloc[:, :39]

train.head()

ps_reg_03_col = train.iloc[:, 22:23]
ps_car_13_col = train.iloc[:, 36:37]

feature_reg03_car13 = ps_reg_03_col.mul(ps_car_13_col['ps_car_13'], axis=0)
feature_reg03_car13.columns = ['feature_car13_mul_reg03']

#ps_reg_03_col_2 = ps_reg_03_col.pow(2, axis='coiumns', level=None, fill_value=None)#increasing power to 2
#ps_reg_03_col_2.columns = ['ps_reg_03_2']
ps_car_13_col_2 = ps_car_13_col.pow(2, axis='coiumns', level=None, fill_value=None)#increasing power to 2
ps_car_13_col_2.columns = ['ps_car_13_2']
train = pd.concat([train, feature_reg03_car13, ps_car_13_col_2], axis=1)

#data processing for test
test.isnull().sum()

imputer_reg_03 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_reg_03.fit(test.iloc[:, 21:22])
test.iloc[:, 21:22] = imputer_reg_03.transform(test.iloc[:, 21:22])

imputer_car_11 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_11.fit(test.iloc[:, 33:34])
test.iloc[:, 33:34] = imputer_car_11.transform(test.iloc[:, 33:34])

imputer_car_12 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_12.fit(test.iloc[:, 34:35])
test.iloc[:, 34:35] = imputer_car_12.transform(test.iloc[:, 34:35])

imputer_car_14 = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
imputer_car_14.fit(test.iloc[:, 36:37])
test.iloc[:, 36:37] = imputer_car_14.transform(test.iloc[:, 36:37])


test.isnull().sum()
#NaN for Cat. add NaN as extra LEVEL

cat_level_values = {'ps_ind_02_cat': 5,'ps_ind_04_cat': 2, 'ps_ind_05_cat': 7, 'ps_car_01_cat': 12, 'ps_car_02_cat': 2,
                    'ps_car_03_cat': 2, 'ps_car_04_cat' : 10, 'ps_car_05_cat' : 2, 'ps_car_06_cat': 18, 'ps_car_07_cat': 2,
                    'ps_car_08_cat': 2, 'ps_car_09_cat': 5, 'ps_car_10_cat': 3, 'ps_car_11_cat': 105}

test.fillna(value=cat_level_values, inplace=True)
test.isnull().sum()
test = test.iloc[:, :38]

ps_reg_03_col_t = test.iloc[:, 21:22]
ps_car_13_col_t = test.iloc[:, 35:36]

feature_reg03_car13_t = ps_reg_03_col_t.mul(ps_car_13_col_t['ps_car_13'], axis=0)
feature_reg03_car13_t.columns = ['feature_car13_mul_reg03']

#ps_reg_03_col_t = test.iloc[:, 21:22]
#ps_reg_03_col_t_2 = ps_reg_03_col_t.pow(2, axis='coiumns', level=None, fill_value=None)#increasing power to 2
#ps_reg_03_col_t_2.columns = ['ps_reg_03_2']

ps_car_13_col_2_t = ps_car_13_col_t.pow(2, axis='coiumns', level=None, fill_value=None)#increasing power to 2
ps_car_13_col_2_t.columns = ['ps_car_13_2']

test = pd.concat([test, feature_reg03_car13_t, ps_car_13_col_2_t], axis=1)

#cat to one-hot-encoder

def process_categorical_features(df):
    dummies_ind_02_cat = pd.get_dummies(df.ps_ind_02_cat, prefix="ps_ind_02_cat", drop_first=True)
    dummies_ind_04_cat = pd.get_dummies(df.ps_ind_04_cat, prefix="ps_ind_04_cat", drop_first=True)
    dummies_ind_05_cat = pd.get_dummies(df.ps_ind_05_cat, prefix="ps_ind_05_cat", drop_first=True)
    dummies_car_01_cat = pd.get_dummies(df.ps_car_01_cat, prefix="ps_car_01_cat", drop_first=True)
    dummies_car_02_cat = pd.get_dummies(df.ps_car_02_cat, prefix="ps_car_02_cat", drop_first=True)
    dummies_car_03_cat = pd.get_dummies(df.ps_car_03_cat, prefix="ps_car_03_cat", drop_first=True)
    dummies_car_04_cat = pd.get_dummies(df.ps_car_04_cat, prefix="ps_car_04_cat", drop_first=True)
    dummies_car_05_cat = pd.get_dummies(df.ps_car_05_cat, prefix="ps_car_05_cat", drop_first=True)
    dummies_car_06_cat = pd.get_dummies(df.ps_car_06_cat, prefix="ps_car_06_cat", drop_first=True)
    dummies_car_07_cat = pd.get_dummies(df.ps_car_07_cat, prefix="ps_car_07_cat", drop_first=True)
    dummies_car_08_cat = pd.get_dummies(df.ps_car_08_cat, prefix="ps_car_08_cat", drop_first=True)
    dummies_car_09_cat = pd.get_dummies(df.ps_car_09_cat, prefix="ps_car_09_cat", drop_first=True)
    dummies_car_10_cat = pd.get_dummies(df.ps_car_10_cat, prefix="ps_car_10_cat", drop_first=True)
    dummies_car_11_cat = pd.get_dummies(df.ps_car_11_cat, prefix="ps_car_11_cat", drop_first=True)
    
    df.drop(["ps_ind_02_cat", "ps_ind_04_cat", "ps_ind_05_cat", "ps_car_01_cat", "ps_car_02_cat", "ps_car_05_cat",
             "ps_car_06_cat", "ps_car_07_cat", "ps_car_08_cat", "ps_car_09_cat", "ps_car_10_cat", "ps_car_11_cat"],
    axis=1, inplace=True)
    
    return pd.concat([df, dummies_ind_02_cat, dummies_ind_04_cat, dummies_ind_05_cat, dummies_car_01_cat,
                     dummies_car_02_cat, dummies_car_03_cat, dummies_car_04_cat, dummies_car_05_cat,
                     dummies_car_06_cat, dummies_car_07_cat, dummies_car_08_cat, dummies_car_09_cat,
                     dummies_car_10_cat, dummies_car_11_cat], axis=1)
    
   # return pd.concat(df,  dummies_car_07_cat, dummies_car_08_cat, dummies_car_09_cat,
     #                dummies_car_10_cat, dummies_car_11_cat)
    
train = process_categorical_features(train)

test = process_categorical_features(test)


#extract data for Train

y = train['target']
X = train[[col for col in train.columns if col != "target"]]
X = X[[col for col in X.columns if col != "id"]]

X.head(5)

#extract data for Test
X_Test_sub = test[[col for col in test.columns if col != "id"]]
Test_id = test[[col for col in test.columns if col == "id"]]



#data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_Test_sub = sc.transform(X_Test_sub)




#model xgboost
xgb_model = xgb.XGBClassifier({'tree_method': 'gpu_hist'})
#dtrain = xgb.DMatrix(X_train, label=y_train)

y_train_matrix = y_train.as_matrix()

sum_wpos = sum( 1 for i in range(len(y_train_matrix)) if y_train_matrix[i] == 1 )
sum_wneg = sum( 1 for i in range(len(y_train_matrix)) if y_train_matrix[i] == 0 )
scale_pos_weight = sum_wneg/sum_wpos

parameters = {
              'objective':['binary:logistic'],
              'learning_rate': [0.01, 0.05], #so called `eta` value
              'max_depth': [5,6, 7],
              'min_child_weight': [11, 15],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [10], #number of trees, change it to 1000 for better results
              
              'seed': [1337],
              'scale_pos_weight': [1, 2, 6, scale_pos_weight],
              'gamma': [1/10.0 for i in range(0, 3)],
              'n_jobs': [-1]
              
              }

#Grid search

grid_search = GridSearchCV(estimator = xgb_model,
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 5, verbose=True)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_

best_accuracy = grid_search.best_score_
rou_auc_gs = best_accuracy
gini_score_gs = 2*best_accuracy -1



#use xgboost API now
xgdmat = xgb.DMatrix(X_train, y_train) #create Dmatrix

new_param = { 'objective':'binary:logistic',
              'gamma': 0.0,
              'learning_rate': 0.01, #so called `eta` value  0.01 is the best
              'max_depth': 7, #previously used 6, 10, 3 is worse
              'min_child_weight': 15, #using 1 no improvement
              'silent': 1,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'n_estimators': 10, #number of trees, change to 100 or  1000 dosn't help
              
              'seed': 1337,
              'scale_pos_weight': 6, #1 is better than formulation of 26.3
              'tree_method': 'gpu_hist'}


cv_xgb = xgb.cv(params = new_param, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['auc'],
                early_stopping_rounds = 100, verbose_eval=True)

cv_xgb.tail(5)

final_gb = xgb.train(new_param, xgdmat, num_boost_round = 766)

#xgb.plot_importance(final_gb)

testdmat = xgb.DMatrix(X_test)

y_pred = final_gb.predict(testdmat)
                


#compute ROC curve, ROC area abd Gini Score
from sklearn.metrics import roc_curve, auc


falsePositiveRate, truePositiveRate, _ = roc_curve(y_test, y_pred)
roc_auc = auc(falsePositiveRate, truePositiveRate)
giniScore_1 = roc_auc*2 -1





#Plot roc curve
plt.figure(1)

plt.plot(falsePositiveRate, truePositiveRate, color='darkorange',
         lw=3, label='ROC curve for claims (area = %0.2f) ' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for XGBoost Model')
plt.legend(loc="lower right")
plt.show()


