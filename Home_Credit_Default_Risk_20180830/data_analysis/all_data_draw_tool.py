#coding=utf-8

"""
Author: Aigege
Code: https://github.com/AiIsBetter
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot
import plotly.plotly as py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from plotly import tools
py.sign_in('Your PlotlyId', 'Your PlotlyKey') # 去plotly注册自己的账号获得key即可.

##################################train and test#################################################################

# def bar_hor(df, col, title, color, w=None, h=None, lm=0, limit=100, return_trace=False, rev=False, xlb = False):
#     cnt_srs = df[col].value_counts()
#     yy = cnt_srs.head(limit).index[::-1]
#     xx = cnt_srs.head(limit).values[::-1]
#     if rev:
#         yy = cnt_srs.tail(limit).index[::-1]
#         xx = cnt_srs.tail(limit).values[::-1]
#     if xlb:
#         trace = go.Bar(y=xlb, x=xx, orientation = 'h', marker=dict(color=color))
#     else:
#         trace = go.Bar(y=yy, x=xx, orientation = 'h', marker=dict(color=color))
#     if return_trace:
#         return trace
#     layout = dict(title=title, margin=dict(l=lm), width=w, height=h)
#     data = [trace]
#     fig = go.Figure(data=data, layout=layout)
#     py.image.save_as(fig,'data_analysis/category/'+col+'_bar_hor.png')
# 
# def exploreCat(col,application):
#     t = application[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/category/' + col + '_exploreCat.png')
#     # plot(fig)
#
# def gp(col, title,application):
#     df1 = application[application["TARGET"] == 1]
#     df0 = application[application["TARGET"] == 0]
#     a1 = df1[col].value_counts()
#     b1 = df0[col].value_counts()
#     total = dict(application[col].value_counts())
#     x0 = a1.index
#     x1 = b1.index
#     y0 = [float(x)*100 / total[x0[i]] for i,x in enumerate(a1.values)]
#     y1 = [float(x)*100 / total[x1[i]] for i,x in enumerate(b1.values)]
#     trace1 = go.Bar(x=a1.index, y=y0, name='Target : 1', marker=dict(color="#96D38C"))
#     trace2 = go.Bar(x=b1.index, y=y1, name='Target : 0', marker=dict(color="#FEBFB3"))
#     return trace1, trace2
# def catAndTrgt(col,application):
#     tr0 = bar_hor(application, col, "Distribution of "+col ,"#f975ae", h=1050,w = 2080, lm=100, return_trace= True)
#     tr1, tr2 = gp(col, 'Distribution of Target with ' + col,application)
#     fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles = [col +" Distribution" , "Target : 1| % Rpyment difficulty by "+col ,"Target : 0|% of otherCases by "+col])
#     fig.append_trace(tr0, 1, 1)
#     fig.append_trace(tr1, 1, 2)
#     fig.append_trace(tr2, 1, 3)
#     fig['layout'].update(height=1050,width = 2080, showlegend=False, margin=dict(l=50))
#     py.image.save_as(fig, 'data_analysis/category/' + col + '_catAndTrgt.png')
#     # plot(fig)
# 
# def numeric(col,application):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(application[col].dropna())
#     plt.savefig('data_analysis/numeric/' + col + '_numeric.png')
#     plt.close()
# application_train = pd.read_csv('../data/application_train.csv')
# print(application_train.info(verbose=True,null_counts=True))
# # application_test= pd.read_csv('../data/application_test.csv')
# count = 0
# judge = 0
# for feature in application_train.columns:
#     if(feature == 'TOTALAREA_MODE'):
#         judge = 1
#         continue
#     if(judge == 0):
#         continue
#     if(application_train[feature].dtypes == 'object'):
#         catAndTrgt(feature, application_train)
#         exploreCat(feature, application_train)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         numeric(feature,application_train)
#         print(str(count)+feature)
# bar_hor(application_train, 'TARGET', '11', ["#96D38C",'#FF4444'], w=1000, h=1000, lm=200, xlb = ['Target:1','Target:0'])
############################################################bureau######################################################################
# def BExpCat(col,bureau):
#     t = bureau[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/bureau_category/' + col + '_cat.png')
#     # iplot(fig)
# def BNumeric(col,bureau):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(bureau[col].dropna())
#     plt.savefig('data_analysis/bureau_numeric/' + col + '_numeric.png')
#     plt.close()
# bureau = pd.read_csv('../data/bureau.csv')
# count = 0
# for feature in bureau.columns:
#     if(bureau[feature].dtypes == 'object'):
#         BExpCat(feature, bureau)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         BNumeric(feature,bureau)
#         print(str(count)+feature)
############################################################bureau_balance######################################################################
# def BBExpCat(col,bureau_balance):
#     t = bureau_balance[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/bureau_balance_category/' + col + '_cat.png')
#     # iplot(fig)
# def BBNumeric(col,bureau_balance):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(bureau_balance[col].dropna())
#     plt.savefig('data_analysis/bureau_balance_numeric/' + col + '_numeric.png')
#     plt.close()
# bureau_balance = pd.read_csv('../data/bureau_balance.csv')
# count = 0
# for feature in bureau_balance.columns:
#     if(bureau_balance[feature].dtypes == 'object'):
#         BBExpCat(feature, bureau_balance)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         BBNumeric(feature,bureau_balance)
#         print(str(count)+feature)
############################################################POS_CASH_balance######################################################################
# def PCExpCat(col,POS_CASH_balance):
#     t = POS_CASH_balance[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/POS_CASH_balance_category/' + col + '_cat.png')
#    # iplot(fig)
# def PCNumeric(col,POS_CASH_balance):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(POS_CASH_balance[col].dropna())
#     plt.savefig('data_analysis/POS_CASH_balance_numeric/' + col + '_numeric.png')
#     plt.close()
# POS_CASH_balance = pd.read_csv("../data/POS_CASH_balance.csv")
# count = 0
# for feature in POS_CASH_balance.columns:
#     if(POS_CASH_balance[feature].dtypes == 'object'):
#         PCExpCat(feature, POS_CASH_balance)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         PCNumeric(feature,POS_CASH_balance)
#         print(str(count)+feature)
############################################################credit_card_balance######################################################################
# def CCExpCat(col,credit_card_balance):
#     t = credit_card_balance[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/credit_card_balance_category/' + col + '_cat.png')
#     # iplot(fig)
# def CCNumeric(col,credit_card_balance):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(credit_card_balance[col].dropna())
#     plt.savefig('data_analysis/credit_card_balance_numeric/' + col + '_numeric.png')
#     plt.close()
# credit_card_balance = pd.read_csv("../data/credit_card_balance.csv")
# count = 0
# for feature in credit_card_balance.columns:
#     if(credit_card_balance[feature].dtypes == 'object'):
#         CCExpCat(feature, credit_card_balance)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         CCNumeric(feature,credit_card_balance)
#         print(str(count)+feature)
############################################################previous_application######################################################################
# def PreExpCat(col,previous_application):
#     t = previous_application[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/previous_application_category/' + col + '_cat.png')
#     # iplot(fig)
# def PreNumeric(col,previous_application):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(previous_application[col].dropna())
#     plt.savefig('data_analysis/previous_application_numeric/' + col + '_numeric.png')
#     plt.close()
# previous_application = pd.read_csv("../data/previous_application.csv")
# count = 0
# for feature in previous_application.columns:
#     if(previous_application[feature].dtypes == 'object'):
#         PreExpCat(feature, previous_application)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         PreNumeric(feature,previous_application)
#         print(str(count)+feature)
############################################################installments_payments######################################################################
# def InsExpCat(col,installments_payments):
#     t = installments_payments[col].value_counts()
#     labels = t.index
#     values = t.values
#     colors = ['#96D38C','#FEBFB3']
#     trace = go.Pie(labels=labels, values=values,
#                    hoverinfo="all", textinfo='value',
#                    textfont=dict(size=12),
#                    marker=dict(colors=colors,
#                                line=dict(color='#fff', width=2)))
#     layout = go.Layout(title=col, height=1050,width = 2080)
#     fig = go.Figure(data=[trace], layout=layout)
#     py.image.save_as(fig, 'data_analysis/installments_payments_category/' + col + '_cat.png')
#     # iplot(fig)
# def InsNumeric(col,installments_payments):
#     plt.figure(figsize=(12,5))
#     plt.title("Distribution of "+col)
#     ax = sns.distplot(installments_payments[col].dropna())
#     plt.savefig('data_analysis/installments_payments_numeric/' + col + '_numeric.png')
#     plt.close()
# installments_payments = pd.read_csv("../data/installments_payments.csv")
# count = 0
# for feature in installments_payments.columns:
#     if(installments_payments[feature].dtypes == 'object'):
#         InsExpCat(feature, installments_payments)
#         count = count +1
#         print(str(count)+feature)
#     else:
#         count = count + 1
#         InsNumeric(feature,installments_payments)
#         print(str(count)+feature)

























