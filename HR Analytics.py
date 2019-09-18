
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from xgboost.sklearn import XGBClassifier, XGBRegressor

pd.set_option('display.max_columns', 500)


# In[70]:


train = pd.read_csv('C:/Users/divya/Downloads/HR analytics/train.csv')
test = pd.read_csv('C:/Users/divya/Downloads/HR analytics/test.csv')
print(train.shape)
print(test.shape)


# In[71]:


train.is_promoted.value_counts()


# In[72]:


all_data = pd.concat([train, test]).reset_index(drop = True)
print(all_data.shape)


# In[73]:


all_data.is_promoted.value_counts(dropna=False)


# In[74]:


np.sum(all_data.isnull())


# In[75]:


## Filling missing enteris with maximum occuring event
all_data['previous_year_rating'].fillna(3.0, inplace=True)
all_data['education'].fillna('Bachelor\'s', inplace=True)


# In[76]:


pyes = train[train['is_promoted']==1]
pno = train[train['is_promoted']==0]


# In[77]:


#calculate target-yes for department
gb = pyes.groupby(['department'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'dept_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['department']).fillna(0)


# In[78]:


#calculate target-no for department
gb = pno.groupby(['department'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'dept_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['department']).fillna(0)


# In[79]:


#calculate target-yes for education
gb = pyes.groupby(['education'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'edu_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['education']).fillna(0)


# In[80]:


#calculate target-no for education
gb = pno.groupby(['education'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'edu_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['education']).fillna(0)


# In[81]:


#calculate target-yes for region
gb = pyes.groupby(['region'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'reg_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['region']).fillna(0)


# In[82]:


#calculate target-no for region
gb = pno.groupby(['region'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'reg_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['region']).fillna(0)


# In[83]:


#calculate target-yes for gender
gb = pyes.groupby(['gender'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'gen_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['gender']).fillna(0)


# In[84]:


#calculate target-no for gender
gb = pno.groupby(['gender'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'gen_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['gender']).fillna(0)


# In[85]:


#calculate target-yes for channel
gb = pyes.groupby(['recruitment_channel'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'rec_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['recruitment_channel']).fillna(0)


# In[86]:


#calculate target-no for channel
gb = pno.groupby(['recruitment_channel'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'rec_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['recruitment_channel']).fillna(0)


# In[87]:


#calculate target-yes for prev
gb = pyes.groupby(['previous_year_rating'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'prev_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['previous_year_rating']).fillna(0)


# In[88]:


#calculate target-no for prev
gb = pno.groupby(['previous_year_rating'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'prev_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['previous_year_rating']).fillna(0)


# In[89]:


#calculate target-yes for awards
gb = pyes.groupby(['awards_won?'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'award_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['awards_won?']).fillna(0)


# In[90]:


#calculate target-no for awards
gb = pno.groupby(['awards_won?'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'award_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['awards_won?']).fillna(0)


# In[91]:


#calculate target-yes for KPI
gb = pyes.groupby(['KPIs_met >80%'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'kpi_target_yes'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['KPIs_met >80%']).fillna(0)


# In[92]:


#calculate target-no for KPI
gb = pno.groupby(['KPIs_met >80%'],as_index=False).agg({'is_promoted':'count'})
gb = gb.rename(columns={'is_promoted':'kpi_target_no'})
# Join it to the all_data
all_data = pd.merge(all_data, gb, how='left', on=['KPIs_met >80%']).fillna(0)


# In[93]:


del gb


# In[94]:


full = all_data.copy()


# In[95]:


all_data = full.copy()


# In[96]:


all_data['cnt_emp_dep'] = all_data.groupby("department")["employee_id"].transform('count')
all_data['avg_dept_score'] = all_data.groupby("department")["avg_training_score"].transform('mean')
all_data['sum_dept_award'] = all_data.groupby("department")["awards_won?"].transform('sum')
all_data['sum_dept_trn'] = all_data.groupby("department")["no_of_trainings"].transform('sum')


# In[97]:


all_data['cnt_emp_edu'] = all_data.groupby("education")["employee_id"].transform('count')
all_data['avg_edu_score'] = all_data.groupby("education")["avg_training_score"].transform('mean')
all_data['sum_edu_award'] = all_data.groupby("education")["awards_won?"].transform('sum')
all_data['sum_edu_trn'] = all_data.groupby("education")["no_of_trainings"].transform('sum')


# In[98]:


def frequency_encoding(df, col_name):
    new_name = "{}_counts".format(col_name)
    new_col_name = "{}_freq".format(col_name)
    grouped = df.groupby(col_name).size().reset_index(name=new_name)
    df = df.merge(grouped, how = "left", on = col_name)
    df[new_col_name] = df[new_name]/df[new_name].count()
    del df[new_name]
    return df


# In[99]:


categorical_features = ['department','education','region','gender','recruitment_channel']

for cat in categorical_features:
        all_data = frequency_encoding(all_data, cat)    


# In[100]:


for col in all_data.columns:
    if all_data[col].nunique() == 1:
        print(col)
        all_data = all_data.drop([col], axis=1)


# In[101]:


#all_data['performance'] = all_data['awards_won?'].apply(str) + "_" + all_data['KPIs_met >80%'].apply(str) + "_" + all_data['previous_year_rating'].apply(str)


# In[102]:


all_data.head()


# In[103]:


cat_feat = ['department','education','region','gender','recruitment_channel']
#convert to dummy variables
all_data = pd.get_dummies(data=all_data,columns=cat_feat) 


# In[104]:


train = all_data.loc[:train.shape[0] - 1,:]
test = all_data.loc[train.shape[0]:,:] 


# In[105]:


print(train.shape)
print(test.shape)


# In[106]:


train.is_promoted.value_counts()


# In[107]:


print(train.shape)
print(test.shape)


# In[108]:


train.is_promoted.value_counts()


# In[109]:


X = train.drop(['employee_id','is_promoted'], axis=1)
Y = train['is_promoted']
X_test = test.drop(['employee_id','is_promoted'], axis=1)


# In[110]:


print(X.shape)
print(Y.shape)
print(X_test.shape)


# In[111]:


n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)


# In[112]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
import lightgbm as lgb
import time
from sklearn.metrics import mean_squared_error, f1_score
from sklearn import linear_model


# In[113]:


def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):
    
    oof = np.zeros(X.shape[0])
    prediction = np.zeros(X_test.shape[0])
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())        
        if model_type == 'sklearn':
            X_train, X_valid = X[train_index], X[valid_index]
        else:
            X_train, X_valid = X.values[train_index], X.values[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators = 200, nthread = 4, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
                    verbose=1000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration_)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
                   
        if model_type == 'xgb':
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test.values)
            
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_squared_error(y_valid, y_pred_valid)
            y_pred = model.predict(X_test)
                    
        
        if model_type == 'cat':
            model = CatBoostClassifier(iterations=200, **params)            
            model.fit(X_train, y_train, 
                      eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(f1_score(y_valid, y_pred_valid) * 100)
        
        prediction += y_pred   
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        
    prediction /= n_fold

    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction


# In[114]:


xgb_params = {'learning_rate':0.1, 
              'n_estimators':200,
              'max_depth':4, 
              'min_child_weight':7, 
              'gamma':0.4,
              'nthread':4, 
              'subsample':0.8, 
              'colsample_bytree':0.8, 
              'objective':'binary:logistic',
               'scale_pos_weight':3,
              'seed':29,
             'silent': True}
oof_xgb, prediction_xgb = train_model(X, X_test, Y, params=xgb_params, model_type='xgb', plot_feature_importance=False)


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_xgb.astype(np.int32)})
submission.to_csv('submission_xgb.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


params = {'num_leaves': 30,
         'min_data_in_leaf': 10,
         'max_depth': 4,
         'learning_rate': 0.1,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1,
          "objective": 'binary',
          "seed":29
         }
oof_lgb, prediction_lgb, _ = train_model(X, X_test, Y, params=params, model_type='lgb', plot_feature_importance=True)      


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_lgb.astype(np.int32)})
submission.to_csv('submission_lgb.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


cat_params = {'learning_rate': 0.1,
              'depth': 5,
              'l2_leaf_reg': 10,
              # 'bootstrap_type': 'Bernoulli',
              'colsample_bylevel': 0.8,
              'bagging_temperature': 0.2,
              #'metric_period': 500,
              'od_type': 'Iter',
              'od_wait': 100,
              'random_seed': 11,
              'allow_writing_files': False}
oof_cat, prediction_cat = train_model(X, X_test, Y, params=cat_params, model_type='cat')


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_cat.astype(np.int32)})
submission.to_csv('submission_cat.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'binary',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'accuracy',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_1, prediction_lgb_1 = train_model(X, X_test, Y, params=params, model_type='lgb', plot_feature_importance=False)


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_lgb_1.astype(np.int32)})
submission.to_csv('submission_lgb_1.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'binary',
         'max_depth': 7,
         'learning_rate': 0.02,
         "boosting": "gbdt",
         "feature_fraction": 0.7,
         "bagging_freq": 5,
         "bagging_fraction": 0.7,
         "bagging_seed": 11,
         "metric": 'accuracy',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_2, prediction_lgb_2 = train_model(X, X_test, Y, params=params, model_type='lgb', plot_feature_importance=False)


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_lgb_2.astype(np.int32)})
submission.to_csv('submission_lgb_2.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


train_stack = np.vstack([oof_lgb, oof_xgb, oof_cat, oof_lgb_1, oof_lgb_2]).transpose()
train_stack = pd.DataFrame(train_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])
test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_cat, prediction_lgb_1, prediction_lgb_2]).transpose()
test_stack = pd.DataFrame(test_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])


# In[ ]:


params = {'num_leaves': 8,
         'min_data_in_leaf': 20,
         'objective': 'binary',
         'max_depth': 2,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_stack, prediction_lgb_stack, _ = train_model(train_stack, test_stack, Y, params=params, model_type='lgb', plot_feature_importance=True)


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_lgb_stack.astype(np.int32)})
submission.to_csv('submission_lgb_stack.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


xgb_params = {'learning_rate':0.1, 
              'n_estimators':200, 
              'max_depth':4, 
              'min_child_weight':7, 
              'gamma':0.4,
              'nthread':4, 
              'subsample':0.8, 
              'colsample_bytree':0.8, 
              'objective':'binary:logistic',
               'scale_pos_weight':3,
              'seed':29,
             'silent': True}
oof_xgb_stack, prediction_xgb_stack = train_model(train_stack, test_stack, Y, params=xgb_params, model_type='xgb', plot_feature_importance=False)


# In[ ]:


submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': prediction_xgb_stack.astype(np.int32)})
submission.to_csv('submission_xgb_stack.csv', index=False)


# In[ ]:


submission.is_promoted.value_counts()


# In[ ]:


blend_lgb_xgb =((prediction_lgb + prediction_xgb) / 2)
submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': blend_lgb_xgb.astype(np.int32)})
submission.to_csv('submission_bl_lgb_xgb.csv', index=False)
submission.is_promoted.value_counts()

#sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat) / 3)
#sub.to_csv("blend_lgb_xgb_cat.csv", index=False)
#sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat + prediction_lgb_1) / 4)
#sub.to_csv("blend_lgb_xgb_cat_lgb1.csv", index=False)
#sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb + prediction_cat + prediction_lgb_1 + prediction_lgb_2) / 5)
#sub.to_csv("blend_lgb_xgb_cat_lgb1_lgb2.csv", index=False)


# In[ ]:


blend_lgb_xgb_cat =((prediction_lgb + prediction_xgb + prediction_cat) / 3)
submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': blend_lgb_xgb_cat.astype(np.int32)})
submission.to_csv('submission_bl_lgb_xgb_cat.csv', index=False)
submission.is_promoted.value_counts()


# In[ ]:


blend_lgb_xgb_cat_lgb1 =((prediction_lgb + prediction_xgb + prediction_cat+prediction_lgb_1) / 4)
submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': blend_lgb_xgb_cat_lgb1.astype(np.int32)})
submission.to_csv('submission_bl_lgb_xgb_cat_lgb1.csv', index=False)
submission.is_promoted.value_counts()


# In[ ]:


blend_lgb_xgb_cat_lgb1_lgb2 =((prediction_lgb + prediction_xgb + prediction_cat+prediction_lgb_1+prediction_lgb_2) / 5)
submission = pd.DataFrame({'employee_id': test['employee_id'].values, 'is_promoted': blend_lgb_xgb_cat_lgb1_lgb2.astype(np.int32)})
submission.to_csv('submission_bl_lgb_xgb_cat_lgb1_lgb2.csv', index=False)
submission.is_promoted.value_counts()

