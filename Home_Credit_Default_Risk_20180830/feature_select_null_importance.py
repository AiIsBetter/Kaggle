#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""
# date 2018.08.29

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb
from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel
import warnings
warnings.simplefilter('ignore', UserWarning)

import gc
gc.enable()
def feature_select_null(path,num_nrows):
    data_all = pd.read_csv(path,nrows = num_nrows)
    file = open('feature_select_name.txt', 'r')
    feature_lines = file.readlines()
    feature_lines = [i.strip('\n') for i in feature_lines]
    file.close()
    data_all = data_all[['SK_ID_CURR', 'TARGET'] + feature_lines]
    data_all = reduce_mem_usage_parallel(data_all, 10)
    data = data_all[data_all.TARGET.notnull()]
    del data_all
    gc.collect()

    def get_feature_importances(data, shuffle, seed=None):
        # 收集真实数据
        train_features = [f for f in data if f not in ['TARGET', 'SK_ID_CURR']]

        y = data['TARGET'].copy()
        if shuffle:
            y = data['TARGET'].copy().sample(frac=1.0)

        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'binary',
            'boosting_type': 'rf',
            'subsample': 0.623,
            'colsample_bytree': 0.7,
            'num_leaves': 127,
            'max_depth': 8,
            'seed': seed,
            'bagging_freq': 1,
            'device' : 'gpu'
        }
        clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['trn_score'] = roc_auc_score(y, clf.predict(data[train_features]))
        return imp_df
    np.random.seed(123)
    actual_imp_df = get_feature_importances(data=data, shuffle=False)
    null_imp_df = pd.DataFrame()
    print("80 started!")
    nb_runs = 80
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        imp_df = get_feature_importances(data=data, shuffle=True)
        imp_df['run'] = i + 1
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
        print(dsp, end='', flush=True)
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    null_imp_df.to_csv('null_importances_distribution_rf.csv')
    actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')
    correlation_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
        gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
        split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
        correlation_scores.append((_f, split_score, gain_score))
    corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])
    def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
        # Fit LightGBM
        dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
        lgb_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'learning_rate': .1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 31,
            'max_depth': -1,
            'seed': 13,
            'device': 'gpu',
            'min_split_gain': .00001,
            'reg_alpha': .00001,
            'reg_lambda': .00001,
            'metric': 'auc'
        }
    # Fit the model
        hist = lgb.cv(
            params=lgb_params,
            train_set=dtrain,
            num_boost_round=2000,
            # categorical_feature=cat_feats,
            nfold=5,
            stratified=True,
            shuffle=True,
            early_stopping_rounds=50,
            verbose_eval=0,
            seed=17
        )
        return hist['auc-mean'][-1], hist['auc-stdv'][-1]
    # features = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
    # score_feature_selection(df=data[features], train_features=features, target=data['TARGET'])
    del data
    gc.collect()
    data_all = pd.read_csv(path,nrows = num_nrows)
    # data_all = reduce_mem_usage(data_all)

    data_all = reduce_mem_usage_parallel(data_all, 10)
    print("select started!")
    for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
        split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
        # split_cat_feats = [_f for _f, _score, _ in correlation_scores if (_score >= threshold)]
        gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
        # gain_cat_feats = [_f for _f, _, _score in correlation_scores if (_score >= threshold) ]

        # print('Results for threshold %3d' % threshold)
        # split_results = score_feature_selection(df=data, train_features=split_feats, cat_feats=None,
        #                                         target=data['TARGET'])
        # print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
        # gain_results = score_feature_selection(df=data, train_features=gain_feats, cat_feats=None,
        #                                        target=data['TARGET'])
        # print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
        split_data = [f for f in data_all.columns if f in ['TARGET', 'SK_ID_CURR'] or f in split_feats]
        gain_data = [f for f in data_all.columns if f  in ['TARGET', 'SK_ID_CURR'] or f in gain_feats ]
        file = open('threshold_'+str(threshold)+ '_split_data.txt', 'w')
        for i in range(len(split_data)):
            file.write(str(split_data[i]) + '\n')
        file.close()

        file = open('threshold_' + str(threshold) + '_gain_data.txt', 'w')
        for i in range(len(gain_data)):
            file.write(str(gain_data[i]) + '\n')
        file.close()
        # data_all[split_data].to_csv('threshold_'+str(threshold)+ '_split_data.csv',index = False)
        # data_all[gain_data].to_csv('threshold_' + str(threshold) + '_gain_data.csv',index = False)
        print(str(threshold)+'_save_finished!!')
    del data_all
    gc.collect()
