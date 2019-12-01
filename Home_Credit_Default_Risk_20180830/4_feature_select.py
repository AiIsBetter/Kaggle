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
from reduce_memory import reduce_mem_usage
from scipy.stats import ranksums
from reduce_memory_parallel import reduce_mem_usage_parallel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def main(debug = False):
    num_rows = 10000 if debug else None
    # 第一次特征筛选
    col_length = []
    for i in range(4132):
        col_length.append(i)
    #电脑只有16g内存，只能分列读取后减少内存再合并了
    if debug:
        df = pd.read_csv('feature_engineering.csv',nrows = num_rows)
        df = reduce_mem_usage_parallel(df, 10)
    else:
        df1 = pd.read_csv('feature_engineering.csv',usecols=col_length[0:2132],nrows = num_rows)
        df1 = reduce_mem_usage_parallel(df1, 10)
        df2 = pd.read_csv('feature_engineering.csv', usecols=col_length[2132:4132],nrows = num_rows)
        df2 = reduce_mem_usage_parallel(df2, 10)
        df = pd.concat([df1,df2],axis = 1)
        del df1,df2
        gc.collect()
        df = reduce_mem_usage_parallel(df, 10)

    def corr_feature_with_target(feature, target,debug = False):
        c0 = feature[target == 0].dropna()
        c1 = feature[target == 1].dropna()
        if set(feature.unique()) == set([0, 1]):
            diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
        else:
            if(debug):
                diff = abs(c0.mean(axis=0) - c1.mean(axis=0))
            else:
                diff = abs(c0.median(axis=0) - c1.median(axis=0))
        # 样本量20以下为小样本情况
        p = ranksums(c0, c1)[1] if ((len(c0) >= 20) & (len(c1) >= 20)) else 2
        return [diff, p]
    nun = df.nunique()
    empty = list(nun[nun <= 1].index)
    print('Before removing empty features there are {0:d} features'.format(df.shape[1]))
    df.drop(empty, axis=1, inplace=True)
    print('After removing empty features there are {0:d} features'.format(df.shape[1]))

    corr = pd.DataFrame(index=['diff', 'p'])
    ind = df[df['TARGET'].notnull()].index
    for c in df.columns.drop('TARGET'):
        corr[c] = corr_feature_with_target(df.loc[ind, c], df.loc[ind, 'TARGET'],debug)
    corr = corr.T
    corr['diff_norm'] = abs(corr['diff'] / df.mean(axis=0))

    to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .05))].index
    to_del_2 = corr[((corr['diff_norm'] < .5) & (corr['p'] > .05))]
    for i in to_del_1 :
        if(i in to_del_2.index):
            to_del_2 = to_del_2.drop(i)

    to_del = list(to_del_1) + list(to_del_2.index)
    if 'SK_ID_CURR' in to_del:
        to_del.remove('SK_ID_CURR')

    df.drop(to_del, axis=1, inplace=True)
    print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(
        df.shape[1]))

    corr_test = pd.DataFrame(index=['diff', 'p'])
    target = df['TARGET'].notnull().astype(int)

    for c in df.columns.drop('TARGET'):
        corr_test[c] = corr_feature_with_target(df[c], target,debug)

    corr_test = corr_test.T
    corr_test['diff_norm'] = abs(corr_test['diff'] / df.mean(axis=0))
    # P = < 0.05，故拒绝原假设下，认为分布有差异
    bad_features = corr_test[((corr_test['p'] < .05) & (corr_test['diff_norm'] > 1))].index
    bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index

    df.drop(bad_features, axis=1, inplace=True)
    print(
        'After removing features with not the same distribution on train and test datasets there are {0:d} features'.format(
            df.shape[1]))
    del corr, corr_test
    gc.collect()

    clf = lgb.LGBMClassifier(random_state=0)
    train_index = df[df['TARGET'].notnull()].index
    train_columns = df.drop('TARGET', axis=1).columns

    score = 1
    new_columns = []
    while score > .7:
        train_columns = train_columns.drop(new_columns)
        clf.fit(df.loc[train_index, train_columns], df.loc[train_index, 'TARGET'])
        f_imp = pd.Series(clf.feature_importances_, index=train_columns)
        score = roc_auc_score(df.loc[train_index, 'TARGET'],
                              clf.predict_proba(df.loc[train_index, train_columns])[:, 1])
        new_columns = f_imp[f_imp > 0].index

    df.drop(train_columns, axis=1, inplace=True)
    print('After removing features not interesting for classifier there are {0:d} features'.format(df.shape[1]))

    df.to_csv('feature_selected.csv', index=False)
if __name__ == "__main__":
    with timer("Full feature select run"):
        main(debug = True)
