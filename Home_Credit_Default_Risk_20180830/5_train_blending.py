#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.08.29

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from feature_select_null_importance import feature_select_null
from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel
from multiprocessing import cpu_count
from scipy.stats import ranksums
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(df, num_folds, stratified=False, debug=False,select_feature = False):

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

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
            n_estimators=5000,
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
            # n_jobs = cpu_count() - 1,
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

    for i in feature_importances:
        feature_importances[i] = feature_importances[i]/5
    feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    if select_feature:
        file = open('feature_importance_select.txt', 'w')
        for i in range(len(feature_importances)):
            file.write(str(feature_importances[i]) + '\n')
        file.close()
    else:
        file = open('feature_importance_noselect.txt', 'w')
        for i in range(len(feature_importances)):
            file.write(str(feature_importances[i]) + '\n')
        file.close()

    if  not debug:
        test_df['TARGET'] = sub_preds
        if select_feature:
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission_Aigege_select.csv', index=False)
        else:
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission_Aigege_noselect.csv', index=False)

    submission = test_df[['SK_ID_CURR', 'TARGET']]
    del test_df,train_df
    gc.collect()
    return feature_importances,feature_drop_no_im,oof_preds,submission

def main(debug = False):
    num_rows = 1000 if debug else None
    with timer("Run LightGBM with kfold blending"):
        feature_select_null('feature_selected.csv',num_rows)
        df = pd.read_csv('feature_selected.csv', nrows=num_rows)
        df = reduce_mem_usage_parallel(df, 10)
        df_train = df[df.TARGET.notnull()]
        df_test = df[df.TARGET.isnull()]
        train_final = df_train[['TARGET','SK_ID_CURR']]
        test_final = df_test[['SK_ID_CURR']]
        del df_test,df_train
        gc.collect()
        app_train = pd.read_csv('prediction_train.csv', nrows=num_rows)
        app_test = pd.read_csv('prediction_test.csv', nrows=num_rows)
        df_temp = pd.concat([app_train, app_test])
        df_temp = reduce_mem_usage_parallel(df_temp, 10)

        for i in range(100):
            if i not in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                continue
            print(i)
            feat_importance_txt = open('threshold_' + str(i) + '_split_data.txt', 'r')
            feature_lines = feat_importance_txt.readlines()
            feature_select = []
            for num, feature in enumerate(feature_lines):
                 temp = feature.strip('\n')
                 feature_select.append(temp)
            feat_importance_txt.close()
            df_select = df[feature_select]
            df_select = pd.merge(df_select, df_temp, how='left', on='SK_ID_CURR')
            feat_importance,feature_drop_no_im,oof_preds,submission = kfold_lightgbm(df_select, num_folds= 5, stratified= False, debug= debug)
            oof_preds = pd.Series(oof_preds)
            oof_preds = pd.DataFrame({'TARGET_split_'+str(i):oof_preds})
            submission.rename(columns={'TARGET':'TARGET_split_'+str(i)},inplace = True)
            submission.drop(['SK_ID_CURR'],axis = 1, inplace = True)
            train_final = pd.concat([train_final,oof_preds],axis = 1)
            test_final = pd.concat([test_final, submission], axis=1)
            del df_select,oof_preds,submission
            gc.collect()
            train_final.to_csv('threshold_'+str(i)+ '_split_train_merge.csv',index=False)
            test_final.to_csv('threshold_' + str(i) + '_split_test_merge.csv', index=False)
        del df,train_final, test_final
        gc.collect()
######################################
        df = pd.read_csv('feature_selected.csv', nrows=num_rows)
        df = reduce_mem_usage_parallel(df, 10)
        df_train = df[df.TARGET.notnull()]
        df_test = df[df.TARGET.isnull()]
        train_final = df_train[['TARGET','SK_ID_CURR']]
        test_final = df_test[['SK_ID_CURR']]
        del df_test,df_train
        gc.collect()
        for i in range(100):
            if i not in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
                continue
            print(i)
            feat_importance_txt = open('threshold_' + str(i) + '_gain_data.txt', 'r')
            feature_lines = feat_importance_txt.readlines()
            feature_select = []
            for num, feature in enumerate(feature_lines):
                 temp = feature.strip('\n')
                 feature_select.append(temp)
            feat_importance_txt.close()
            df_select = df[feature_select]
            feat_importance,feature_drop_no_im,oof_preds,submission = kfold_lightgbm(df_select, num_folds= 5, stratified= False, debug= debug)
            oof_preds = pd.Series(oof_preds)
            oof_preds = pd.DataFrame({'TARGET_gain_'+str(i):oof_preds})
            submission.rename(columns={'TARGET':'TARGET_gain_'+str(i)},inplace = True)
            submission.drop(['SK_ID_CURR'],axis = 1, inplace = True)
            train_final = pd.concat([train_final,oof_preds],axis = 1)
            test_final = pd.concat([test_final, submission], axis=1)
            del df_select,oof_preds,submission
            gc.collect()
            train_final.to_csv('threshold_'+str(i)+ '_gain_train_merge.csv',index=False)
            test_final.to_csv('threshold_' + str(i) + '_gain_test_merge.csv', index=False)
        del df,train_final, test_final
        gc.collect()

if __name__ == "__main__":
    with timer("Full blending run"):
        main()
