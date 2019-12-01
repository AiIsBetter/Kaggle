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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis, iqr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import sqrt
import sys
from tqdm import tqdm, trange
from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel
from multiprocessing import cpu_count
from scipy.stats import ranksums
from src.utils import parallel_apply
from src.feature_extraction import add_features_in_group
from functools import partial
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# stacking
def get_oof(clf,train_x1,train_y1,test_x1,test_y1,num_folds,fillna = False):

    if fillna:
        train_x = train_x1.replace([np.inf,-np.inf],0)
        train_x.fillna(0, inplace=True)
        test_x = test_x1.replace([np.inf,-np.inf], 0)
        test_x.fillna(0, inplace=True)
    else:
        train_x = train_x1
        test_x = test_x1

    oof_train = np.zeros((train_x.shape[0],))
    oof_test = np.zeros((test_x.shape[0],))
    oof_test_skf = np.empty((num_folds, test_x.shape[0]))
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    for i, (train_index, test_index) in enumerate(tqdm(folds.split(train_x),desc='stacking-cv',file=sys.stdout,total = num_folds)):
        x_tr = train_x.loc[train_index]
        y_tr = train_y1.loc[train_index]
        x_te = train_x.loc[test_index]
        y_te = train_y1.loc[test_index]
        clf.train(x_tr, y_tr,x_te,y_te)
        # 存入交叉验证结果
        oof_train[test_index] = clf.predict(x_te)
        #存入测试集结果
        oof_test_skf[i, :] = clf.predict(test_x)
    # 测试集结果取均值保存为一列，每个模型一列
    oof_test[:] = oof_test_skf.mean(axis=0)
    del train_x,test_x
    gc.collect()
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
    def train(self, train_x, train_y, valid_x, valid_y):
        self.clf.fit(train_x, train_y)
    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_seed'] = seed
        self.clf = clf(**params)
    def train(self, train_x, train_y, valid_x, valid_y):
        self.clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                     verbose=100, early_stopping_rounds=200)
    def predict(self, x):
        return self.clf.predict_proba(x)[:, 1]

class LightGBMWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['feature_fraction_seed'] = seed
        params['bagging_seed'] = seed
        self.clf = clf(**params)
    def train(self, train_x, train_y,valid_x,valid_y):
        self.clf.fit(train_x, train_y,eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)
    def predict(self, x):
        return self.clf.predict_proba(x, num_iteration=self.clf.best_iteration_)[:, 1]

class XgbWrapper(object):
    def __init__(self,clf, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.clf = clf(**params)
    def train(self, train_x, train_y,valid_x,valid_y):
        self.clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                     eval_metric='auc', verbose=100, early_stopping_rounds=200)
    def predict(self, x):
        return self.clf.predict_proba(x, ntree_limit=self.clf.best_ntree_limit)[:, 1]

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
            boosting_type = 'gbdt',
            objective = 'binary',
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves = 30,
            max_bin = 250,
            max_depth =  -1,
            min_child_samples =70,
            subsample =1.0,
            subsample_freq = 1,
            colsample_bytree =0.05,
            min_gain_to_split = 0.5,
            reg_lambda = 100.0,
            reg_alpha = 0.0,
            scale_pos_weight = 1,
            is_unbalance = False,
            device='gpu'
        )
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=100)

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

    if not debug:
        test_df['TARGET'] = sub_preds
        if select_feature:
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission_Aigege_select.csv', index=False)
        else:
            test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission_Aigege_noselect.csv', index=False)

    return feature_importances,feature_drop_no_im
def main(debug = False):
    num_rows = 100000 if debug else None
    df = pd.read_csv('feature_selected.csv',nrows = num_rows)
    df = reduce_mem_usage_parallel(df, 10)

    file = open('feature_select_name.txt', 'r')
    feature_lines = file.readlines()
    feature_lines =[i.strip('\n') for i in feature_lines]
    file.close()
    df = df[['SK_ID_CURR','TARGET']+feature_lines]

    app_train = pd.read_csv('prediction_train.csv', nrows=num_rows)
    app_test = pd.read_csv('prediction_test.csv', nrows=num_rows)
    df_temp = pd.concat([app_train, app_test])

    df_temp = reduce_mem_usage_parallel(df_temp, 10)
    df = pd.merge(df, df_temp, how='left', on='SK_ID_CURR')

    df = reduce_mem_usage_parallel(df, 10)
    with timer("Run LightGBM with kfold and single model output"):
        feat_importance,feature_drop_no_im = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)
 #########################################选出前n个特征以及交叉特征，开始stacking############################################################
    with timer("Run Stacking"):
        et_params = {
            'n_jobs': 16,
            'n_estimators':800,
            'max_features': 0.5,
            'max_depth': 12,
            'min_samples_leaf': 2,
            'n_jobs': cpu_count() - 1
        }

        rf_params = {
            'n_jobs': 16,
            'n_estimators': 800,
            'max_features': 0.2,
            'max_depth': 12,
            'min_samples_leaf': 2,
            'n_jobs':cpu_count() - 1
        }

        xgb_params = {
            'seed': 0,
            'n_estimators':10000,
            'colsample_bytree': 0.7,
            'silent': 1,
            'subsample': 0.7,
            'learning_rate': 0.045,
            'objective': 'binary:logistic',
            'max_depth': 4,
            'num_parallel_tree': 1,
            'min_child_weight': 1,
            'nrounds': 200,

            'tree_method':'gpu_hist'
        }

        catboost_params = {
            'iterations': 10000,
            'learning_rate': 0.02,
            'depth': 3,
            'l2_leaf_reg': 40,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.7,
            'scale_pos_weight': 5,
            'eval_metric': 'AUC',
            'od_type': 'Iter',
            'allow_writing_files': False
        }

        lightgbm_params = {
            'nthread': 4,
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight':  39.3259775,
            'silent' : -1,
            'verbose': -1,
            'device': 'gpu'
        }
        SEED = 0
        # stacking结果
        xg = XgbWrapper(clf = xgb.XGBClassifier,seed=SEED, params=xgb_params)
        et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
        rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
        cb = CatboostWrapper(clf=CatBoostClassifier, seed=SEED, params=catboost_params)
        lg = LightGBMWrapper(clf=lgb.LGBMClassifier, seed=SEED, params=lightgbm_params)

        train = df[df['TARGET'].notnull()]
        test = df[df['TARGET'].isnull()]
        del df
        gc.collect()
        train_y = train['TARGET']
        test_y = test['TARGET']
        del  train['TARGET'],test['TARGET']
        gc.collect()

        excluded_feats = ['SK_ID_CURR']
        features = [f_ for f_ in train.columns if f_ not in excluded_feats]

        train_x = train[features]
        test_x = test[features]

        print('xg started!')
        xg_oof_train, xg_oof_test = get_oof(xg, train_x, train_y, test_x, test_y, num_folds=5)
        print('et started!')
        et_oof_train, et_oof_test = get_oof(et,train_x,train_y,test_x,test_y,num_folds=5,fillna = True)
        print('rf started!')
        rf_oof_train, rf_oof_test = get_oof(rf,train_x,train_y,test_x,test_y,num_folds=5,fillna = True)
        print('cb started!')
        cb_oof_train, cb_oof_test = get_oof(cb,train_x,train_y,test_x,test_y,num_folds=5,fillna = True)
        print('lg started!')
        lg_oof_train, lg_oof_test = get_oof(lg, train_x, train_y, test_x, test_y,num_folds=5)
        print('all finished!')

        print("ET-CV: {}".format(roc_auc_score(train_y, et_oof_train)))
        print("RF-CV: {}".format(roc_auc_score(train_y, rf_oof_train)))
        print("XG-CV: {}".format(roc_auc_score(train_y, xg_oof_train)))
        print("cb-CV: {}".format(roc_auc_score(train_y, cb_oof_train)))
        print("LG-CV: {}".format(roc_auc_score(train_y, lg_oof_train)))
        del train_x,test_x
        gc.collect()
        train_x = np.concatenate((et_oof_train, rf_oof_train, cb_oof_train,lg_oof_train), axis=1)
        test_x = np.concatenate((et_oof_test, rf_oof_test, cb_oof_test,lg_oof_test), axis=1)
        # train_x = np.concatenate((xg_oof_train, cb_oof_train,lg_oof_train), axis=1)
        # test_x = np.concatenate((xg_oof_test,cb_oof_test,lg_oof_test), axis=1)

        np.save('train_x.npy',train_x)
        np.save('test_x.npy', test_x)
        np.save('train_y.npy', train_y)

        print("{},{}".format(train_x.shape, test_x.shape))
        print('xgb merge started!')

        # xgbstacking
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x)
        del train_x,test_x,xg_oof_train, cb_oof_train,lg_oof_train,xg_oof_test,cb_oof_test,lg_oof_test
        gc.collect()
        xgb_params = {
            'seed': 0,
            'colsample_bytree': 0.8,
            'silent': 1,
            'subsample': 0.6,
            'learning_rate': 0.01,
            'objective': 'reg:linear',
            'max_depth': 4,
            'num_parallel_tree': 1,
            'min_child_weight': 1,
            'eval_metric': 'auc',
            'tree_method': 'gpu_hist'
        }
        print("xgb cv..")
        folds = KFold(n_splits=5, shuffle=True, random_state=1001)
        res = xgb.cv(xgb_params, dtrain, num_boost_round=5000, nfold=5, folds = folds,seed=SEED, stratified=False,
                     early_stopping_rounds=100, verbose_eval=100, show_stdv=True)
        best_nrounds = res.shape[0] - 1
        print("meta xgb train..")
        gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
        fi = gbdt.predict(dtest)
        fi = np.array(fi)
        test['TARGET'] = fi
        # logistic_regression = LogisticRegression()
        # logistic_regression.fit(train_x, train_y)
        #
        # test['TARGET'] = logistic_regression.predict_proba(test_x)[:, 1]
        test[['SK_ID_CURR', 'TARGET']].to_csv('stacking_submission.csv', index=False, float_format='%.8f')
        print('save finished!')
        # feature_importances = LGB_test(train,test)
if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()
