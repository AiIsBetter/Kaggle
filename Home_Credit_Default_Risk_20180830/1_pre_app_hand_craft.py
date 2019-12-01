#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.08.29

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve

import gc
import time
from tqdm import tqdm, trange
from contextlib import contextmanager
from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def one_hot_encoder(data, nan_as_category=True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns \
                           if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace=True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis=1, inplace=True)
    return data, [c for c in data.columns if c not in original_columns]

def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # kfold交叉验证
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2008)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=2008)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    from multiprocessing import cpu_count
    feature_importances = dict(zip(feats,np.zeros(train_df.columns.shape[0])))
    feature_drop_no_im = dict(zip(feats,np.zeros(train_df.columns.shape[0])))
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        # clf = lgb.LGBMClassifier(
        #     nthread=4,
        #     n_estimators=10000,
        #     learning_rate=0.02,
        #     num_leaves=34,
        #     colsample_bytree=0.9497036,
        #     subsample=0.8715623,
        #     max_depth=8,
        #     reg_alpha=0.041545473,
        #     reg_lambda=0.0735294,
        #     min_split_gain=0.0222415,
        #     min_child_weight=39.3259775,
        #     silent=-1,
        #     verbose=-1,
        #     # n_jobs = cpu_count() - 1,
        #     device = 'gpu'
        # )
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=30,
            max_bin=250,
            max_depth=-1,
            min_child_samples=70,
            subsample=1.0,
            subsample_freq=1,
            colsample_bytree=0.05,
            min_gain_to_split=0.5,
            reg_lambda=100.0,
            reg_alpha=0.0,
            scale_pos_weight=1,
            is_unbalance=False,
            device='gpu'
        )
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        feature_importances1 = dict(zip(train_x.columns, clf.feature_importances_))

        for i in feature_importances1:
            feature_importances[i] = feature_importances[i] +feature_importances1[i]
            if(feature_importances1[i] ==0):
                feature_drop_no_im[i] = feature_drop_no_im[i]+1
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    test_df['prediction'] = sub_preds
    train_df['prediction'] = oof_preds
    df = pd.concat([train_df,test_df])
    df = df.drop(['TARGET'],axis = 1)


    del test_df,train_df
    gc.collect()
    return df
def target_prediction_features(df_train,df_test,path,num_rows,name_id,nan_as_category = True):
    df_preds = pd.read_csv(path, nrows=num_rows)
    df_preds = df_preds.merge(df_train[['SK_ID_CURR','TARGET']],on = 'SK_ID_CURR' , how = 'left')
    df_preds,_ = one_hot_encoder(df_preds, nan_as_category)
    df_preds = reduce_mem_usage_parallel(df_preds, 10)
    df_preds = kfold_lightgbm(df_preds, 5, stratified=False, debug=False)
    df_preds = df_preds[['SK_ID_CURR', 'prediction']]
    agg_prev_score = df_preds.groupby('SK_ID_CURR', as_index=False)['prediction'].agg(
        {name_id+'_prediction_mean': 'mean', name_id+'_prediction_max': 'max', name_id+'_prediction_sum': 'sum',
         name_id+'_prediction_min': 'min', name_id+'_prediction_count': 'median', name_id+'_prediction_std': 'std'})
    df_train = df_train.merge(agg_prev_score, on='SK_ID_CURR', how='left')
    df_test = df_test.merge( agg_prev_score, on='SK_ID_CURR', how='left')
    del df_preds,agg_prev_score
    gc.collect()
    return df_train,df_test
def main(debug = False):
    num_rows = 100000 if debug else None

    with timer("Process  previous_applications prediction train:"):
        df_train = pd.read_csv('../../data/application_train.csv', nrows=num_rows)
        df_test = pd.read_csv('../../data/application_test.csv', nrows=num_rows)
        original_col = [i for i in df_train.columns if i != 'SK_ID_CURR' and  i != 'TARGET']
        print('ccb started!')
        path = '../../data/credit_card_balance.csv'
        df_train,df_test = target_prediction_features(df_train,df_test,path,num_rows,'ccb')
        print('ins started!')
        path = '../../data/installments_payments.csv'
        df_train, df_test = target_prediction_features(df_train, df_test, path, num_rows, 'ins')
        print('pos started!')
        path = '../../data/POS_CASH_balance.csv'
        df_train, df_test = target_prediction_features(df_train, df_test, path, num_rows, 'pos')
        print('pre started!')
        path = '../../data/previous_application.csv'
        df_train, df_test = target_prediction_features(df_train, df_test, path, num_rows, 'pre')
        print('bur started!')
        path = '../../data/bureau.csv'
        df_train, df_test = target_prediction_features(df_train, df_test, path, num_rows, 'bur')
        df_train.drop(original_col,axis = 1,inplace = True)
        df_train.drop('TARGET', axis=1, inplace=True)
        df_test.drop(original_col, axis=1, inplace=True)
        df_train.to_csv('prediction_train.csv',index = False)
        df_test.to_csv('prediction_test.csv', index=False)
if __name__ == "__main__":
    with timer("Full feature engineering run") as a:
        main()
