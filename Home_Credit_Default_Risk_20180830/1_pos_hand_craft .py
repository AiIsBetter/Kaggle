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
from src.utils import parallel_apply
from src.feature_extraction import add_features_in_group
from functools import partial

from reduce_memory import reduce_mem_usage
from reduce_memory_parallel import reduce_mem_usage_parallel

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# 最后k分期付款统计特征
def last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                         ['count', 'mean'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                         ['count', 'mean'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'min', 'median'],
                                         period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'min', 'median'],
                                         period_name)
    return features
# 最后一笔贷款统计特征：
def last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

    features = {}
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                     ['count', 'sum', 'mean'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                     ['sum', 'mean'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                     ['sum', 'mean', 'max', 'min', 'std'],
                                     'last_loan_')
    return features
# 最后k笔分期付款趋势特征
def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_trend_feature(features, gr_period,
                                     'SK_DPD', '{}_period_trend_'.format(period)
                                     )
        features = add_trend_feature(features, gr_period,
                                     'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                     )
    return features
# 趋势特征趋势计算
def add_trend_feature(features, gr, feature_name, prefix):
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
    num_worker = 10
    df_train = pd.read_csv('../../data/application_train.csv',nrows = num_nrows)
    df_test = pd.read_csv('../../data/application_test.csv',nrows = num_nrows)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    original_col = [i for i in df_train if i != 'TARGET' and i != 'SK_ID_CURR']
    df = reduce_mem_usage_parallel(df,10)

    app_train = df[df.TARGET.notnull()]
    app_test = df[df.TARGET.isnull()]
    pos_cash_balance = pd.read_csv('../../data/POS_CASH_balance.csv', nrows=num_nrows)
    pos_cash_balance = reduce_mem_usage_parallel(pos_cash_balance,10)


    features = pd.DataFrame({'SK_ID_CURR': pos_cash_balance['SK_ID_CURR'].unique()})
    pos_cash_sorted = pos_cash_balance.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    group_object = pos_cash_sorted.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].last().reset_index()
    group_object.rename(index=str,
                        columns={'CNT_INSTALMENT_FUTURE': 'pos_cash_remaining_installments'},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    pos_cash_balance['is_contract_status_completed'] = pos_cash_balance['NAME_CONTRACT_STATUS'] == 'Completed'
    group_object = pos_cash_balance.groupby(['SK_ID_CURR'])['is_contract_status_completed'].sum().reset_index()
    group_object.rename(index=str,
                        columns={'is_contract_status_completed': 'pos_cash_completed_contracts'},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')


    pos_cash_balance['pos_cash_paid_late'] = (pos_cash_balance['SK_DPD'] > 0).astype(int)
    pos_cash_balance['pos_cash_paid_late_with_tolerance'] = (pos_cash_balance['SK_DPD_DEF'] > 0).astype(int)
    groupby = pos_cash_balance.groupby(['SK_ID_CURR'])
    func = partial(last_k_installment_features, periods=[1, 10, 50, 10e16])

    g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=num_worker, chunk_size=10000).reset_index()
    features = features.merge(g, on='SK_ID_CURR', how='left')

    g = parallel_apply(groupby, last_loan_features, index_name='SK_ID_CURR', num_workers=num_worker,
                       chunk_size=10000).reset_index()
    features = features.merge(g, on='SK_ID_CURR', how='left')

    func = partial(trend_in_last_k_installment_features, periods=[1, 6, 12, 30, 60])
    g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=num_worker, chunk_size=10000).reset_index()
    features = features.merge(g, on='SK_ID_CURR', how='left')

    app_train = app_train.merge(features, on='SK_ID_CURR', how='left')
    app_test = app_test.merge(features, on='SK_ID_CURR', how='left')
    app_train.drop(original_col,axis =1,inplace = True)
    app_test.drop(original_col, axis=1, inplace=True)
    app_train.to_csv('pos_hand_crafted_train.csv',index = False)
    app_test.to_csv('pos_hand_crafted_test.csv', index=False)
