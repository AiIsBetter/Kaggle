#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.08.29

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization

df_train = pd.read_csv('threshold_99_split_train_merge.csv', index_col = 'SK_ID_CURR')
df_test = pd.read_csv('threshold_99_split_test_merge.csv')

for c in df_train.columns.drop('TARGET'):
    print(c, roc_auc_score(df_train['TARGET'], df_train[c]))

def ROC_evaluate(**params):
    warnings.simplefilter('ignore')
    s = sum(params.values())
    for p in params:
        params[p] = params[p] / s
    test_pred_proba = pd.Series(np.zeros(df_train.shape[0]), index=df_train.index)
    feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]
    for f in feats:
        test_pred_proba += df_train[f] * params[f]
    return roc_auc_score(df_train['TARGET'], test_pred_proba)
params = {}
for c in df_train.columns.drop('TARGET'):
    params[c] = (0, 1)
bo = BayesianOptimization(ROC_evaluate, params)
bo.maximize(init_points=200, n_iter=20)
best_params = bo.res['max']['max_params']
print(bo.res['max']['max_val'])
best_normalized_params = {}
s = sum(best_params.values())
for p in best_params:
    best_normalized_params[p] = best_params[p] / s

prediction_train = pd.Series(np.zeros(df_train.shape[0]), index=df_train.index)
feats = [f for f in df_train.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]
for f in feats:
    prediction_train += df_train[f] * best_normalized_params[f]
print(roc_auc_score(df_train['TARGET'], prediction_train))

prediction_test = pd.Series(np.zeros(df_test.shape[0]), index=df_test.index)
feats = [f for f in df_test.columns if f not in ['TARGET', 'SK_ID_CURR', 'index']]
for f in feats:
    prediction_test += df_test[f] * best_normalized_params[f]
submission = pd.Series(prediction_test)
submission = pd.DataFrame({'TARGET':submission})
submission = pd.concat([df_test[['SK_ID_CURR']],submission],axis = 1)
submission.to_csv('submission_blend.csv',index = False)
