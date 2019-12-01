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
from sklearn.linear_model import LinearRegression
from scipy.stats import skew, kurtosis, iqr
from src.utils import parallel_apply
from src.feature_extraction import add_features_in_group
from functools import partial
from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def add_features(feature_name, aggs, features, feature_names, groupby):
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])

    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg

        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                     columns={
                                                                         feature_name: '{}_{}'.format(feature_name,
                                                                                                      agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]
        # 形成要传入计算的agg函数的名字并计算对应的agg值
        features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_late',
                                         ['count', 'mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_over',
                                         ['count', 'mean'],
                                         'last_{}_'.format(period))
    return features
def last_k_instalment_features_with_fractions(gr, periods, fraction_periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))

        features = add_features_in_group(features, gr_period, 'instalment_paid_late_in_days',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_late',
                                         ['count', 'mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_over_amount',
                                         ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features, gr_period, 'instalment_paid_over',
                                         ['count', 'mean'],
                                         'last_{}_'.format(period))

    for short_period, long_period in fraction_periods:
        short_feature_names = _get_feature_names(features, short_period)
        long_feature_names = _get_feature_names(features, long_period)

        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
    return pd.Series(features)

def _get_feature_names(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])

def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0

def trend_in_last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    if (gr.shape[0]>10):
        a = 1
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)

    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        features = _add_trend_feature(features, gr_period,
                                      'instalment_paid_late_in_days', '{}_period_trend_'.format(period)
                                      )
        features = _add_trend_feature(features, gr_period,
                                      'instalment_paid_over_amount', '{}_period_trend_'.format(period)
                                      )
    return features

def _add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features
if __name__ == '__main__':
    num_nrows = 10000
    num_workers = 10
    df_train = pd.read_csv('../../data/application_train.csv',nrows = num_nrows)
    df_test = pd.read_csv('../../data/application_test.csv',nrows = num_nrows)
    original_col = [i for i in df_train if i !='TARGET' and i !='SK_ID_CURR']
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df = reduce_mem_usage_parallel(df,10)

    app_train = df[df.TARGET.notnull()]
    app_test = df[df.TARGET.isnull()]
    installments = pd.read_csv('../../data/installments_payments.csv',nrows = num_nrows)
    installments_ = installments.sort_values(['SK_ID_PREV']).reset_index(drop = True)
    installments = reduce_mem_usage_parallel(installments,10)

    installments_ = installments
    installments_['instalment_paid_late_in_days'] = installments_['DAYS_ENTRY_PAYMENT'] - installments_[
        'DAYS_INSTALMENT']
    installments_['instalment_paid_late'] = (installments_['instalment_paid_late_in_days'] > 0).astype(int)
    installments_['instalment_paid_over_amount'] = installments_['AMT_PAYMENT'] - installments_['AMT_INSTALMENT']
    installments_['instalment_paid_over'] = (installments_['instalment_paid_over_amount'] > 0).astype(int)

    features = pd.DataFrame({'SK_ID_CURR': installments_['SK_ID_CURR'].unique()})
    groupby = installments_.groupby(['SK_ID_CURR'])
    feature_names = []
    features, feature_names = add_features('NUM_INSTALMENT_VERSION',
                                           ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                           features, feature_names, groupby)

    features, feature_names = add_features('instalment_paid_late_in_days',
                                           ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                           features, feature_names, groupby)

    features, feature_names = add_features('instalment_paid_late', ['sum', 'mean'],
                                           features, feature_names, groupby)

    features, feature_names = add_features('instalment_paid_over_amount',
                                           ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'],
                                           features, feature_names, groupby)

    features, feature_names = add_features('instalment_paid_over', ['sum', 'mean'],
                                           features, feature_names, groupby)

    func = partial(last_k_instalment_features, periods=[1, 5, 10, 20, 50, 100])

    g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                       num_workers=num_workers, chunk_size=10000)
    features = features.merge(g, on='SK_ID_CURR', how='left')

    # 上述新特征的最后k笔分期付款的趋势统计特征，既用线性回归做斜率统计，所以k值选择了较大的几个数值
    func = partial(trend_in_last_k_instalment_features, periods=[10, 50, 100, 500])

    g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                       num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='SK_ID_CURR', how='left')

    # 上述新特征的最后k笔分期付款的统计特征
    func = partial(last_k_instalment_features_with_fractions,
                   periods=[1, 5, 10, 20, 50, 100],
                   fraction_periods=[(5, 20), (5, 50), (10, 100)])

    g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                       num_workers=num_workers, chunk_size=10000).reset_index()
    features = features.merge(g, on='SK_ID_CURR', how='left')

    app_train = app_train.merge(features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(features, on='SK_ID_CURR', how='left')
    app_train.drop(original_col,axis =1,inplace = True)
    app_test.drop(original_col, axis=1, inplace=True)
    app_train.to_csv('ins_hand_crafted_train.csv',index = False)
    app_test.to_csv('ins_hand_crafted_test.csv', index=False)


