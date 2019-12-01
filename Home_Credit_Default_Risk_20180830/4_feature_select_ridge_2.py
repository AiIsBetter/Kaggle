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
from sklearn.linear_model import Ridge

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

def ridge_feature_select(df, num_folds, stratified=False, debug=False,select_feature = False):

    train_df = df
    print("Starting feature select. Train shape: {}".format(train_df.shape))

    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2008)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=2008)

    oof_preds = np.zeros(train_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    feature_importances = dict(zip(feats,np.zeros(train_df.columns.shape[0])))
    feature_drop_no_im = dict(zip(feats,np.zeros(train_df.columns.shape[0])))

    kfold_auc = []
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        clf = Ridge(alpha=1)
        clf.fit(train_x, train_y)
        oof_preds[valid_idx] = clf.predict(valid_x)
        kfold_auc.append('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    full_auc =  roc_auc_score(train_df['TARGET'], oof_preds)
    del train_df
    gc.collect()
    return kfold_auc,full_auc
# 用ridge筛选特征，特征从2000+减少到600+，且pb有差不多0.0001提升
def main(debug = False):
    num_rows = 1000 if debug else None
    with timer("Run ridge select feature:"):
        df = pd.read_csv('feature_selected.csv',nrows = num_rows)

        df = df[df['TARGET'].notnull()]
        df.replace([np.inf, -np.inf], np.nan,inplace = True)
        df.fillna(0, inplace=True)
        df = reduce_mem_usage_parallel(df, 10)

        feat_importance_txt = open('feature_importance_lgb_final.txt', 'r')
        feature_lines = feat_importance_txt.readlines()
        feature_select = []
        for num, feature in enumerate(feature_lines):
             temp = feature.rfind(',')
             temp = feature[0:temp]
             temp = temp[2:-1]
             feature_select.append(temp)
        feat_importance_txt.close()
        feature_array = []
        full_auc_all = 0
        count = 0
        for fea in feature_select:
            print(count)
            count = count + 1
            feature_array.append(fea)
            df_select = df[['SK_ID_CURR','TARGET']+feature_array]
            kfold_auc, full_auc = ridge_feature_select(df_select, num_folds= 5, stratified= False, debug= debug)

            if full_auc_all>=full_auc:
                feature_array.remove(fea)
            else:
                full_auc_all = full_auc
                file = open('feature_select_name.txt', 'a')
                file.write(fea + '\n')
                file.close()
                file = open('feature_select_fullauc.txt', 'a')
                a = str(full_auc_all)
                file.write(str(full_auc_all) + '\n')
                file.close()
                file = open('feature_select_kfoldauc.txt', 'a')
                file.write(str(kfold_auc) + '\n')
                file.close()
            del df_select
            gc.collect()

if __name__ == "__main__":
    with timer("Full feature select run"):
        main()
