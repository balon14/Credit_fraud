#!/usr/bin/env python
# coding: utf-8

# # Импорт библиотек и загрузка данных

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import optuna
import os
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[20]:


BASE_DIR = Path(os.getcwd())

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# In[21]:


train

# In[22]:


train.info()

# In[18]:


train.isnull().sum()

# In[19]:


train.drop(["id"], axis=1, inplace=True)

# In[20]:


train.describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T

# In[21]:


train.duplicated().sum()

# In[22]:


feature_number = (train.columns.to_list())
len(feature_number)
feature_number

# In[24]:


row,col = 10,3
fig,axis = plt.subplots(row, col, figsize=(20,40))
fig.tight_layout(pad=5.0)
col_num = 0
for i in range(row):
    for j in range(col):
        if(col_num > len(feature_number)-2):
            break
        sns.histplot(ax=axis[i,j], x=train[feature_number[col_num]], kde=True)
        axis[i,j].set_xlabel(feature_number[col_num], fontsize=14)
        axis[i,j].set_ylabel('Count', fontsize=14)
        axis[i,j].set_title(f'Distribution of { feature_number[col_num] }', fontsize=16)
        col_num += 1

# In[26]:


#generating time features
train["day"] = (train["Time"]/ (3600 *24)).round(0)
train["hour"] = ((train["Time"]/3600)%24).round(0)

# In[27]:


fraud = train[train["Class"] == 1]

fig,axis = plt.subplots(figsize=(20,5))
sns.countplot(x=fraud["hour"], ax=axis)

# In[28]:


#Checking For Correlation
fig, axis= plt.subplots(figsize=(40,40))
sns.heatmap(train.corr(), annot=True, ax=axis)

# In[29]:


def across_col_feat(df):

    features = [feat for feat in df.columns if 'V' in feat]
    df['V_Sum'] = df[features].sum(axis = 1)
    df['V_Min'] = df[features].min(axis = 1)
    df['V_Max'] = df[features].max(axis = 1)
    df['V_Avg'] = df[features].mean(axis = 1)
    df['V_Std'] = df[features].std(axis = 1)
    df['V_Var'] = df[features].var(axis = 1)
    
    return df
train = across_col_feat(train)
test  = across_col_feat(test)

# In[30]:


X = train.drop(["Time","Class","id"], axis=1, errors="ignore")
y = train["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# In[31]:


#calculating class weights
weights = pd.DataFrame(compute_sample_weight('balanced', train['Class']), columns=['weight'])

#features to scale
scaleFeatures = X.columns.to_list()
rs = RobustScaler()
X[scaleFeatures] = rs.fit_transform(X[scaleFeatures])
X.head()

# In[32]:


SEED = 0
#Defining Objective function
skfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0) 

def objective_xgb(eg):
    
    params = {
        "verbosity":0,
        "objective":"binary:logistic",
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        "eval_metric":"auc",
        "lambda":eg.suggest_float("lambda", 0.01, 10),
        "alpha":eg.suggest_float("alpha", 0.01, 10),
        "max_depth": eg.suggest_int("max_depth", 1, 20),
        "subsample": eg.suggest_categorical('subsample', [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
        "colsample_bytree": eg.suggest_categorical('colsample_bytree', [0.2, 0.4, 0.6, 0.8, 1.0]),
        "learning_rate": eg.suggest_categorical("learning_rate", [0.01, 0.02, 0.03, 0.05, 0.07, 0.09, 0.1]),
        "n_estimators": eg.suggest_int("n_estimators",100,2000),
        "early_stopping":100,
        "random_state":SEED,
        "sample_weight":weights
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    scores = cross_validate(model, X, y, cv=skfold, scoring="roc_auc")
    return roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    return score

# In[33]:


#Generating Study
study = optuna.create_study(direction="maximize")
optuna.logging.set_verbosity(optuna.logging.INFO)
#study.optimize(objective_xgb, n_trials=20)

# In[36]:


#display(study.best_value)
#display(study.best_params)

# In[38]:


#Training XGBoost with best params
best_params = {'lambda': 3.3505139360015495,
 'alpha': 6.403269527191655,
 'max_depth': 11,
 'subsample': 0.8,
 'colsample_bytree': 0.2,
 'learning_rate': 0.05,
 'n_estimators': 366}

gpu_params = {
    "objective":"binary:logistic",
    'tree_method': 'hist', 
    'device': 'cuda',
    'predictor': 'gpu_predictor',
    "eval_metric":"auc",
    "sample_weight":weights
}

params =  {**best_params, **gpu_params}

model = XGBClassifier(**params)
model.fit(X_train, y_train)
folds = 5
cv_results =  cross_validate(model, X_train, y_train, scoring='roc_auc', cv=skfold)
print(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# # Test

# In[40]:


X_test = test.drop(["id","Time"], axis=1, errors="ignore")
X_test["day"] = (test["Time"]/ (3600 *24)).round(0)
X_test["hour"] = ((test["Time"]/3600)%24).round(0)
X_test  = across_col_feat(X_test)
X_test.head()

# In[41]:


X_test[scaleFeatures] = rs.transform(X_test[scaleFeatures])
X_test
