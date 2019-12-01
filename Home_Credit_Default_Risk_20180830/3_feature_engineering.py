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
def rename(df,add_name,on):
    temp = []
    for e in df.columns.tolist():
        if (e[0].find(on) == 0):
            temp.append(e[0])
        else:
            if add_name == None:
                temp.append(e[0] + "_" + e[1].upper())
            else:
                temp.append(add_name + e[0] + "_" + e[1].upper())
    df.columns = pd.Index(temp)
    return df

def one_hot_encoder(data, nan_as_category = True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns \
                           if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis = 1, inplace = True)
    return data, [c for c in data.columns if c not in original_columns]

def application_train_test(num_rows = None, nan_as_category = False):

    df_train = pd.read_csv('../../data/application_train.csv', nrows= num_rows)
    df_test = pd.read_csv('../../data/application_test.csv', nrows= num_rows)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    gc.collect()

    df.drop(df[df['CODE_GENDER'] == 'XNA'].index, inplace=True)
    df.drop(df[df['NAME_INCOME_TYPE'] == 'Maternity leave'].index, inplace=True)
    df.drop(df[df['NAME_FAMILY_STATUS'] == 'Unknown'].index, inplace=True)


    df.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14',
             'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
             'FLAG_DOCUMENT_21'], axis=1, inplace=True)


    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df.loc[df['OWN_CAR_AGE'] > 80, 'OWN_CAR_AGE'] = np.nan
    df.loc[df['REGION_RATING_CLIENT_W_CITY'] < 0, 'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[df['AMT_INCOME_TOTAL'] > 1e8, 'AMT_INCOME_TOTAL'] = np.nan
    df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT'] > 10, 'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
    df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE'] > 40, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan


    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], _ = pd.factorize(df[bin_feature])

    df['app missing'] = df.isnull().sum(axis=1).values

    df['app EXT_SOURCE mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['app EXT_SOURCE std'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['app EXT_SOURCE prod'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * EXT_SOURCE_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['app EXT_SOURCE_1 * EXT_SOURCE_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_2 * EXT_SOURCE_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_1 * DAYS_EMPLOYED'] = df['EXT_SOURCE_1'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_2 * DAYS_EMPLOYED'] = df['EXT_SOURCE_2'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_3 * DAYS_EMPLOYED'] = df['EXT_SOURCE_3'] * df['DAYS_EMPLOYED']
    df['app EXT_SOURCE_1 / DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_2 / DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
    df['app EXT_SOURCE_3 / DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']

     # 20180827新增
    app_train = df[df.TARGET.notnull()]
    app_test = df[df.TARGET.isnull()]
    app_train['external_sources_weighted'] = app_train['EXT_SOURCE_1'] * 2 + app_train['EXT_SOURCE_2'] * 3 + app_train['EXT_SOURCE_3'] * 4
    app_test['external_sources_weighted'] = app_test['EXT_SOURCE_1'] * 2 + app_test['EXT_SOURCE_2'] * 3 + app_test['EXT_SOURCE_3'] * 4

    for function_name in ['min', 'max', 'sum', 'nanmedian']:
        app_train['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
        app_test['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    AGGREGATION_RECIPIES = [
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                                  ('AMT_CREDIT', 'max'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('OWN_CAR_AGE', 'max'),
                                                  ('OWN_CAR_AGE', 'sum')]),
        (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                ('AMT_INCOME_TOTAL', 'mean'),
                                                ('DAYS_REGISTRATION', 'mean'),
                                                ('EXT_SOURCE_1', 'mean')]),
        (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                     ('CNT_CHILDREN', 'mean'),
                                                     ('DAYS_ID_PUBLISH', 'mean')]),
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                               ('EXT_SOURCE_2',
                                                                                                'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                      ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                      ('APARTMENTS_AVG', 'mean'),
                                                      ('BASEMENTAREA_AVG', 'mean'),
                                                      ('EXT_SOURCE_1', 'mean'),
                                                      ('EXT_SOURCE_2', 'mean'),
                                                      ('EXT_SOURCE_3', 'mean'),
                                                      ('NONLIVINGAREA_AVG', 'mean'),
                                                      ('OWN_CAR_AGE', 'mean'),
                                                      ('YEARS_BUILD_AVG', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                ('EXT_SOURCE_1', 'mean')]),
        (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                               ('CNT_CHILDREN', 'mean'),
                               ('CNT_FAM_MEMBERS', 'mean'),
                               ('DAYS_BIRTH', 'mean'),
                               ('DAYS_EMPLOYED', 'mean'),
                               ('DAYS_ID_PUBLISH', 'mean'),
                               ('DAYS_REGISTRATION', 'mean'),
                               ('EXT_SOURCE_1', 'mean'),
                               ('EXT_SOURCE_2', 'mean'),
                               ('EXT_SOURCE_3', 'mean')]),
    ]
    groupby_aggregate_names = []
    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        group_object = app_train.groupby(groupby_cols)
        group_object_test = app_test.groupby(groupby_cols)
        for select, agg in tqdm(specs):
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            app_train = app_train.merge(group_object[select]
                        .agg(agg)
                        .reset_index()
                        .rename(index=str,
                                columns={select: groupby_aggregate_name})
                        [groupby_cols + [groupby_aggregate_name]],
                        on=groupby_cols,
                        how='left')
            app_test = app_test.merge(group_object_test[select]
                                        .agg(agg)
                                        .reset_index()
                                        .rename(index=str,
                                                columns={select: groupby_aggregate_name})
                                        [groupby_cols + [groupby_aggregate_name]],
                                        on=groupby_cols,
                                        how='left')
            groupby_aggregate_names.append(groupby_aggregate_name)

    diff_feature_names = []
    for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
        for select, agg in tqdm(specs):
            if agg in ['mean', 'median', 'max', 'min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
                diff_name = '{}_diff'.format(groupby_aggregate_name)
                abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

                app_train[diff_name] = app_train[select] - app_train[groupby_aggregate_name]
                app_test[diff_name] = app_test[select] - app_test[groupby_aggregate_name]

                app_train[abs_diff_name] = np.abs(app_train[select] - app_train[groupby_aggregate_name])
                app_test[abs_diff_name] = np.abs(app_test[select] - app_test[groupby_aggregate_name])

    app_train['long_employment'] = (app_train['DAYS_EMPLOYED'] < -2000).astype(int)
    app_train['retirement_age'] = (app_train['DAYS_BIRTH'] < -14000).astype(int)
    app_train['cnt_non_child'] = app_train['CNT_FAM_MEMBERS'] - app_train['CNT_CHILDREN']
    app_train['child_to_non_child_ratio'] = app_train['CNT_CHILDREN'] / app_train['cnt_non_child']
    app_train['income_per_non_child'] = app_train['AMT_INCOME_TOTAL'] / app_train['cnt_non_child']
    app_train['credit_per_person'] = app_train['AMT_CREDIT'] / app_train['CNT_FAM_MEMBERS']
    app_train['credit_per_child'] = app_train['AMT_CREDIT'] / (1 + app_train['CNT_CHILDREN'])
    app_train['credit_per_non_child'] = app_train['AMT_CREDIT'] / app_train['cnt_non_child']

    app_test['long_employment'] = (app_test['DAYS_EMPLOYED'] < -2000).astype(int)
    app_test['retirement_age'] = (app_test['DAYS_BIRTH'] < -14000).astype(int)
    app_test['cnt_non_child'] = app_test['CNT_FAM_MEMBERS'] - app_test['CNT_CHILDREN']
    app_test['child_to_non_child_ratio'] = app_test['CNT_CHILDREN'] / app_test['cnt_non_child']
    app_test['income_per_non_child'] = app_test['AMT_INCOME_TOTAL'] / app_test['cnt_non_child']
    app_test['credit_per_person'] = app_test['AMT_CREDIT'] / app_test['CNT_FAM_MEMBERS']
    app_test['credit_per_child'] = app_test['AMT_CREDIT'] / (1 + app_test['CNT_CHILDREN'])
    app_test['credit_per_non_child'] = app_test['AMT_CREDIT'] / app_test['cnt_non_child']
    df = pd.concat([app_train,app_test])
# ###################################################################################
    df, _ = one_hot_encoder(df, nan_as_category)
    df['app AMT_CREDIT - AMT_GOODS_PRICE'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['app AMT_CREDIT / AMT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['app AMT_CREDIT / AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['app AMT_CREDIT / AMT_INCOME_TOTAL'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    df['app AMT_INCOME_TOTAL / 12 - AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / 12. - df['AMT_ANNUITY']
    df['app AMT_INCOME_TOTAL / AMT_ANNUITY'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    df['app AMT_INCOME_TOTAL - AMT_GOODS_PRICE'] = df['AMT_INCOME_TOTAL'] - df['AMT_GOODS_PRICE']
    df['app AMT_INCOME_TOTAL / CNT_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['app AMT_INCOME_TOTAL / CNT_CHILDREN'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    df['app most popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
        .isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
    df['app popular AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'] \
        .isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})

    df['app OWN_CAR_AGE / DAYS_BIRTH'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['app OWN_CAR_AGE / DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']

    df['app DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['app DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['app DAYS_EMPLOYED - DAYS_BIRTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['app DAYS_EMPLOYED / DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df['app CNT_CHILDREN / CNT_FAM_MEMBERS'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    df_bureau_b = pd.read_csv('../../data/bureau_balance.csv', nrows=num_rows)

    tmp = df_bureau_b[['SK_ID_BUREAU', 'STATUS']].groupby('SK_ID_BUREAU')
    tmp_last = tmp.last()
    tmp_last.columns = ['First_status']
    df_bureau_b = df_bureau_b.join(tmp_last, how='left', on='SK_ID_BUREAU')
    tmp_first = tmp.first()
    tmp_first.columns = ['Last_status']
    df_bureau_b = df_bureau_b.join(tmp_first, how='left', on='SK_ID_BUREAU')
    del tmp, tmp_first, tmp_last
    gc.collect()

    tmp = df_bureau_b[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['Month']
    df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
    del tmp
    gc.collect()

    tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == 'C', ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
        .groupby('SK_ID_BUREAU').last()
    tmp = tmp.apply(abs)
    tmp.columns = ['When_closed']
    df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
    del tmp
    gc.collect()

    df_bureau_b['Month_closed_to_end'] = df_bureau_b['Month'] - df_bureau_b['When_closed']

    for c in range(6):
        tmp = df_bureau_b.loc[df_bureau_b['STATUS'] == str(c), ['SK_ID_BUREAU', 'MONTHS_BALANCE']] \
            .groupby('SK_ID_BUREAU').count()
        tmp.columns = ['DPD_' + str(c) + '_cnt']
        df_bureau_b = df_bureau_b.join(tmp, how='left', on='SK_ID_BUREAU')
        df_bureau_b['DPD_' + str(c) + ' / Month'] = df_bureau_b['DPD_' + str(c) + '_cnt'] / df_bureau_b['Month']
        del tmp
        gc.collect()
    df_bureau_b['Non_zero_DPD_cnt'] = df_bureau_b[
        ['DPD_1_cnt', 'DPD_2_cnt', 'DPD_3_cnt', 'DPD_4_cnt', 'DPD_5_cnt']].sum(axis=1)

    df_bureau_b, bureau_b_cat = one_hot_encoder(df_bureau_b, nan_as_category)

    aggregations = {}
    for col in df_bureau_b.columns:
        aggregations[col] = ['mean'] if col in bureau_b_cat else ['min', 'max', 'size']
    df_bureau_b_agg = df_bureau_b.groupby('SK_ID_BUREAU').agg(aggregations)
    df_bureau_b_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in df_bureau_b_agg.columns.tolist()])
    del df_bureau_b
    gc.collect()

    df_bureau = pd.read_csv('../../data/bureau.csv', nrows=num_rows)

    df_bureau.loc[df_bureau['AMT_ANNUITY'] > .8e8, 'AMT_ANNUITY'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_SUM'] > 3e8, 'AMT_CREDIT_SUM'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_SUM_DEBT'] > 1e8, 'AMT_CREDIT_SUM_DEBT'] = np.nan
    df_bureau.loc[df_bureau['AMT_CREDIT_MAX_OVERDUE'] > .8e8, 'AMT_CREDIT_MAX_OVERDUE'] = np.nan
    df_bureau.loc[df_bureau['DAYS_ENDDATE_FACT'] < -10000, 'DAYS_ENDDATE_FACT'] = np.nan
    df_bureau.loc[(df_bureau['DAYS_CREDIT_UPDATE'] > 0) | (
    df_bureau['DAYS_CREDIT_UPDATE'] < -40000), 'DAYS_CREDIT_UPDATE'] = np.nan
    df_bureau.loc[df_bureau['DAYS_CREDIT_ENDDATE'] < -10000, 'DAYS_CREDIT_ENDDATE'] = np.nan
    df_bureau.drop(df_bureau[df_bureau['DAYS_ENDDATE_FACT'] < df_bureau['DAYS_CREDIT']].index, inplace=True)


    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_DEBT']
    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_LIMIT']
    df_bureau['bureau AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = df_bureau['AMT_CREDIT_SUM'] - df_bureau[
        'AMT_CREDIT_SUM_OVERDUE']

    df_bureau['bureau DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = df_bureau['DAYS_CREDIT'] - df_bureau['CREDIT_DAY_OVERDUE']
    df_bureau['bureau DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_CREDIT_ENDDATE']
    df_bureau['bureau DAYS_CREDIT - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT'] - df_bureau['DAYS_ENDDATE_FACT']
    df_bureau['bureau DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = df_bureau['DAYS_CREDIT_ENDDATE'] - df_bureau[
        'DAYS_ENDDATE_FACT']
    df_bureau['bureau DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = df_bureau['DAYS_CREDIT_UPDATE'] - df_bureau[
        'DAYS_CREDIT_ENDDATE']

    df_bureau, bureau_cat = one_hot_encoder(df_bureau, nan_as_category)

    df_bureau = df_bureau.join(df_bureau_b_agg, how='left', on='SK_ID_BUREAU')
    df_bureau.drop('SK_ID_BUREAU', axis=1, inplace=True)
    del df_bureau_b_agg
    gc.collect()


    categorical = bureau_cat + bureau_b_cat
    aggregations = {}
    for col in df_bureau.columns:
        aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_bureau_agg = df_bureau.groupby('SK_ID_CURR').agg(aggregations)
    df_bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in df_bureau_agg.columns.tolist()])


    active_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Active'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(active_agg, how='left')
    del active_agg
    gc.collect()


    closed_agg = df_bureau[df_bureau['CREDIT_ACTIVE_Closed'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    df_bureau_agg = df_bureau_agg.join(closed_agg, how='left')
    del closed_agg, df_bureau
    gc.collect()

    return df_bureau_agg

def previous_applications(num_rows = None, nan_as_category = True):
    df_prev = pd.read_csv('../../data/previous_application.csv', nrows = num_rows)

    df_prev.loc[df_prev['AMT_CREDIT'] > 6000000, 'AMT_CREDIT'] = np.nan
    df_prev.loc[df_prev['SELLERPLACE_AREA'] > 3500000, 'SELLERPLACE_AREA'] = np.nan
    df_prev[['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
             'DAYS_LAST_DUE', 'DAYS_TERMINATION']].replace(365243, np.nan, inplace=True)


    df_prev['prev missing'] = df_prev.isnull().sum(axis=1).values
    df_prev['prev AMT_APPLICATION / AMT_CREDIT'] = df_prev['AMT_APPLICATION'] / df_prev['AMT_CREDIT']
    df_prev['prev AMT_APPLICATION - AMT_CREDIT'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_CREDIT']
    df_prev['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = df_prev['AMT_APPLICATION'] - df_prev['AMT_GOODS_PRICE']
    df_prev['prev AMT_GOODS_PRICE - AMT_CREDIT'] = df_prev['AMT_GOODS_PRICE'] - df_prev['AMT_CREDIT']
    df_prev['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = df_prev['DAYS_FIRST_DRAWING'] - df_prev['DAYS_FIRST_DUE']
    df_prev['prev DAYS_TERMINATION less -500'] = (df_prev['DAYS_TERMINATION'] < -500).astype(int)


    df_prev, categorical = one_hot_encoder(df_prev, nan_as_category)


    aggregations = {}
    for col in df_prev.columns:
        aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_prev_agg = df_prev.groupby('SK_ID_CURR').agg(aggregations)
    df_prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in df_prev_agg.columns.tolist()])


    approved_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Approved'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(approved_agg, how='left')
    del approved_agg
    gc.collect()


    refused_agg = df_prev[df_prev['NAME_CONTRACT_STATUS_Refused'] == 1].groupby('SK_ID_CURR').agg(aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    df_prev_agg = df_prev_agg.join(refused_agg, how='left')
    del refused_agg, df_prev
    gc.collect()

    return df_prev_agg

def previous_applications_hand_crafted(df,num_rows = None):
    app_train = pd.read_csv('../../data/application_train.csv', nrows=num_rows)
    app_test = pd.read_csv('../../data/application_test.csv', nrows=num_rows)
    df_temp = pd.concat([app_train,app_test])
    df_temp = reduce_mem_usage(df_temp)
    original_col = [i for i in df_temp.columns if  i != 'SK_ID_CURR']
    del app_test,app_train
    gc.collect()
    previous_application = pd.read_csv('../../data/previous_application.csv', nrows = num_rows)
    numbers_of_applications = [1, 3, 5]

    features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
    prev_applications_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

    group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
    group_object.rename(index=str,
                        columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    prev_applications_sorted['previous_application_prev_was_approved'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
    group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
        'previous_application_prev_was_approved'].last().reset_index()
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    prev_applications_sorted['previous_application_prev_was_refused'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
    group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
        'previous_application_prev_was_refused'].last().reset_index()
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    for number in numbers_of_applications:
        prev_applications_tail = prev_applications_sorted.groupby(by=['SK_ID_CURR']).tail(number)

        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['CNT_PAYMENT'].mean().reset_index()
        group_object.rename(index=str, columns={
            'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_DECISION'].mean().reset_index()
        group_object.rename(index=str, columns={
            'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

        group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_FIRST_DRAWING'].mean().reset_index()
        group_object.rename(index=str, columns={
            'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)},
                            inplace=True)
        features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
    df_temp = df_temp.merge(features,
                                    left_on=['SK_ID_CURR'],
                                    right_on=['SK_ID_CURR'],
                                    how='left',
                                    validate='one_to_one')

    df_temp.drop(original_col,axis = 1, inplace = True)
    df = pd.merge(df,df_temp,how = 'left',on = 'SK_ID_CURR')
    del df_temp
    gc.collect()
    return df

def pos_cash(num_rows = None, nan_as_category = True):
    df_pos = pd.read_csv('../../data/POS_CASH_balance.csv', nrows = num_rows)


    df_pos.loc[df_pos['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan

    df_pos['pos CNT_INSTALMENT more CNT_INSTALMENT_FUTURE'] = \
        (df_pos['CNT_INSTALMENT'] > df_pos['CNT_INSTALMENT_FUTURE']).astype(int)


    df_pos, categorical = one_hot_encoder(df_pos, nan_as_category)

    aggregations = {}
    for col in df_pos.columns:
        aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_pos_agg = df_pos.groupby('SK_ID_CURR').agg(aggregations)
    df_pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in df_pos_agg.columns.tolist()])


    df_pos_agg['POS_COUNT'] = df_pos.groupby('SK_ID_CURR').size()
    del df_pos
    gc.collect()
    return df_pos_agg


def installments_payments(num_rows = None, nan_as_category = True):
    df_ins = pd.read_csv('../../data/installments_payments.csv', nrows = num_rows)

    df_ins.loc[df_ins['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
    df_ins.loc[df_ins['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan

    df_ins['ins NUM_INSTALMENT_NUMBER_100'] = (df_ins['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
    df_ins['ins DAYS_INSTALMENT more NUM_INSTALMENT_NUMBER'] = (
    df_ins['DAYS_INSTALMENT'] > df_ins['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)
    df_ins['ins AMT_PAYMENT / AMT_INSTALMENT'] = df_ins['AMT_PAYMENT'] / df_ins['AMT_INSTALMENT']

    df_ins, categorical = one_hot_encoder(df_ins, nan_as_category)

    aggregations = {}
    for col in df_ins.columns:
        aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_ins_agg = df_ins.groupby('SK_ID_CURR').agg(aggregations)
    df_ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in df_ins_agg.columns.tolist()])

    df_ins_agg['INSTAL_COUNT'] = df_ins.groupby('SK_ID_CURR').size()
    del df_ins
    gc.collect()
    return df_ins_agg

def credit_card_balance(num_rows = None, nan_as_category = True):
    df_card = pd.read_csv('../../data/credit_card_balance.csv', nrows = num_rows)

    df_card.loc[df_card['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
    df_card.loc[df_card['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan

    df_card['card missing'] = df_card.isnull().sum(axis=1).values
    df_card['card SK_DPD - MONTHS_BALANCE'] = df_card['SK_DPD'] - df_card['MONTHS_BALANCE']
    df_card['card SK_DPD_DEF - MONTHS_BALANCE'] = df_card['SK_DPD_DEF'] - df_card['MONTHS_BALANCE']
    df_card['card SK_DPD - SK_DPD_DEF'] = df_card['SK_DPD'] - df_card['SK_DPD_DEF']

    df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card['AMT_RECIVABLE']
    df_card['card AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_TOTAL_RECEIVABLE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']
    df_card['card AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_RECIVABLE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']

    df_card['card AMT_BALANCE - AMT_RECIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_RECIVABLE']
    df_card['card AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = df_card['AMT_BALANCE'] - df_card[
        'AMT_RECEIVABLE_PRINCIPAL']
    df_card['card AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = df_card['AMT_BALANCE'] - df_card['AMT_TOTAL_RECEIVABLE']

    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_ATM_CURRENT']
    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_OTHER_CURRENT']
    df_card['card AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = df_card['AMT_DRAWINGS_CURRENT'] - df_card[
        'AMT_DRAWINGS_POS_CURRENT']

    df_card, categorical = one_hot_encoder(df_card, nan_as_category)

    aggregations = {}
    for col in df_card.columns:
        aggregations[col] = ['mean'] if col in categorical else ['min', 'max', 'size', 'mean', 'var', 'sum']
    df_card_agg = df_card.groupby('SK_ID_CURR').agg(aggregations)
    df_card_agg.columns = pd.Index(['CARD_' + e[0] + "_" + e[1].upper() for e in df_card_agg.columns.tolist()])

    df_card_agg['CARD_COUNT'] = df_card.groupby('SK_ID_CURR').size()
    del df_card
    gc.collect()
    return df_card_agg

def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = pd.merge(df,bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = pd.merge(df, prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process previous_applications hand crafted"):
        df = previous_applications_hand_crafted(df,num_rows)
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = pd.merge(df, pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = pd.merge(df, ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = pd.merge(df, cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()

    df = reduce_mem_usage_parallel(df,10)
    with timer("Merge hand craft feature"):
        app_train = pd.read_csv('pos_hand_crafted_train.csv',nrows = num_rows)
        app_test = pd.read_csv('pos_hand_crafted_test.csv',nrows = num_rows)
        df_temp = pd.concat([app_train,app_test])
        df_temp = reduce_mem_usage_parallel(df_temp, 10)

        df = pd.merge(df,df_temp,how = 'left',on = 'SK_ID_CURR')
        app_train = pd.read_csv('ins_hand_crafted_train.csv',nrows = num_rows)
        app_test = pd.read_csv('ins_hand_crafted_test.csv',nrows = num_rows)
        df_temp = pd.concat([app_train, app_test])
        df_temp = reduce_mem_usage_parallel(df_temp, 10)

        df = pd.merge(df, df_temp, how='left', on='SK_ID_CURR')
        df.drop(['TARGET_x','TARGET_y'],axis = 1,inplace = True)
        del app_train,app_test,df_temp
        gc.collect()

    df.to_csv('feature_engineering.csv', index=False)
if __name__ == "__main__":
    with timer("Full feature engineering run") as a:
        main(debug = True)
