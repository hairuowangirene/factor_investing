#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

import statsmodels.api as sm


def returns(df, columns, n=1):
    returns = pd.Series(
        np.log(df[f'{columns}'] / (df[f'{columns}'].shift(n) + 0.0001)),
        name=f'returns_{columns}_' + str(n))
    df = df.join(returns)
    return df


def ts_delta(df, columns, n):
    delta = pd.Series(df[f'{columns}'] - df[f'{columns}'].shift(n),
                      name='ts_delta_' + str(n))
    df = df.join(delta)
    return df


def ts_delay(df, columns, n):
    ts_delay = pd.Series(df[f'{columns}'].shift(n), name='ts_delay_' + str(n))
    df = df.join(ts_delay)
    return df


def ts_mean(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    mean = pd.Series(df[f'{columns}'].rolling(n).mean(),
                     name='ts_mean_' + str(n))
    df = df.join(mean)
    return df


def ts_sum(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    sum = pd.Series(df[f'{columns}'].rolling(n).sum(), name='ts_sum_' + str(n))
    df = df.join(sum)
    return df


def ts_std(df, columns, n):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    std = pd.Series(df[f'{columns}'].rolling(n).std(),
                    name=f'ts_std_{columns}_' + str(n))
    df = df.join(std)
    return df


def ts_min(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    min = pd.Series(df[f'{columns}'].rolling(n).min(), name='ts_min_' + str(n))
    df = df.join(min)
    return df


def ts_max(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    max = pd.Series(df[f'{columns}'].rolling(n).max(), name='ts_max_' + str(n))
    df = df.join(max)
    return df


def ts_mean_delta(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    mean = pd.Series(df[f'{columns}'].rolling(n).mean())
    ts_mean_delta = pd.Series(df[f'{columns}'] - mean,
                              name='ts_mean_delta_' + str(n))
    df = df.join(ts_mean_delta)
    return df


def ts_mean_ratio(df, columns, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    mean = pd.Series(df[f'{columns}'].rolling(n).mean())
    ts_mean_ratio = pd.Series(df[f'{columns}'] / (mean + 0.0001),
                              name='ts_mean_ratio_' + str(n))
    df = df.join(ts_mean_ratio)
    return df


def ts_corr(df, columns_1, columns_2, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    corr = pd.Series(df[f'{columns_1}'].rolling(n).corr(df[f'{columns_2}']),
                     name='ts_corr_' + str(n))
    corr = corr.ffill()
    df = df.join(corr)
    return df


def ts_cov(df, columns_1, columns_2, n):
    """

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ts_cov = pd.Series(df[f'{columns_1}'].rolling(n).cov(df[f'{columns_2}']),
                       name='ts_cov_' + str(n))
    df = df.join(ts_cov)
    return df


def ts_decay_linear(df, columns, n):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ts_decay_linear = df[f'{columns}'].copy()
    w = 1
    for i in range(1, n):
        ts_decay_linear_0 = df[f'{columns}'].copy()
        ts_decay_linear += ts_decay_linear_0.shift(n - 1) * i / n
        w += i / n
    ts_decay_linear = pd.Series(ts_decay_linear / w,
                                name='ts_decay_linear_' + str(n))
    df = df.join(ts_decay_linear)
    return df


def ts_decay_exp_window(df, columns, n, factor=0.5):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    ts_decay_exp_window = df[f'{columns}'].copy()
    w = 1
    for i in range(1, n + 1):
        ts_decay_0 = df[f'{columns}'].copy()
        ts_decay_exp_window += ts_decay_0.shift(n - i) * (factor ** (n - 1))
        w += factor ** (n - 1)
    ts_decay_exp_window = pd.Series(ts_decay_exp_window / w,
                                    name='ts_decay_exp_window_' + str(n))
    df = df.join(ts_decay_exp_window)
    return df


def ts_rank(df, columns, n):
    rank = pd.Series(df[f'{columns}'].rolling(n).apply(
        lambda x: n - np.array(x).argsort().argsort()[-1]),
                     name=f'ts_rank_' + str(n))
    df = df.join(rank)
    return df


def ts_zscore(df, columns, n):
    MA = pd.Series(df[f'{columns}'].rolling(n, min_periods=n).mean())
    MSD = pd.Series(df[f'{columns}'].rolling(n, min_periods=n).std())
    ts_zscore = pd.Series((df[f'{columns}'] - MA) / (MSD + 0.0001),
                          name=f'ts_zscore_{columns}_' + str(n))
    df = df.join(ts_zscore)
    return df


def vwap(df, n=5):
    """
    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    M = df['Close'] * df['Volume']
    N = df['Volume'].rolling(n).sum()
    vwap = pd.Series(M.rolling(n).sum() / N, name='vwap_' + str(n))
    df = df.join(vwap)
    return df


def OLS(df, columns_y, columns_x1, columns_xn):
    x = df.loc[:, columns_x1:columns_xn]
    y = df.loc[:, columns_y]
    model = sm.OLS(y, x, hasconst=False)  # 生成模型
    result = model.fit()
    last = np.array(y) - np.array(result.fittedvalues)
    return last[-1]


def ts_neutralize(df, columns_y, columns_x1, columns_xn, n):
    ts_neutralize = [np.nan for i in range(n - 1)]
    for i in range(len(df) - n + 1):
        data = df.iloc[i:n + i, :]
        ts_neutralize.append([OLS(data, columns_y, columns_x1, columns_xn)][0])
    neutralize = pd.Series(ts_neutralize, name='ts_neutralize_' + str(n))
    neutralize.reset_index(drop=True, inplace=True)
    df = df.join(neutralize)
    return df

def sma(df,columns,n, m=2):
    df = df.fillna(0)
    sma = pd.Series((df[f'{columns}'] * m + df[f'{columns}'].shift(1) *
                     (n - m)) / float(n), name=f'sma_{columns}_' + str(n)+str(m)
                    )
    df = df.join(sma)
    return df

def wma(df,columns,n):
    w = np.arange(1, n+1) * 0.9
    w = w/w.sum()
    wma=pd.Series(df[f'{columns}'].rolling(n).apply(lambda x: (x * w).sum()), name=f'wma_{columns}_' + str(n))
    df = df.join(wma)
    return df

def sequence(n):
    return np.arange(1, n + 1)

def regbeta(df,columns, B, n=None):
    if n == None:
        n = len(B)
    rb = pd.Series(df[f'{columns}'].rolling(window=n, center=False)
                   .apply(lambda x: np.cov(x, B)[0][1] / np.var(B)),
                   name=f'regbeta_{columns}_' + str(n))
    df = df.join(rb)
    return df

def lowday(df, columns,window=10):
    ld = pd.Series((window - 1) - df[f'{columns}'].
                   rolling(window).apply(lambda x: np.argsort(x)[0]),
                   name=f'lowday_{columns}_' + str(window))
    df = df.join(ld)
    return df

def highday(df,columns,window=10):
    hd = pd.Series((window - 1) - df[f'{columns}'].
                   rolling(window).apply(lambda x: np.argsort(x)[window - 1]),
                   name=f'highday_{columns}_' + str(window))
    df = df.join(hd)
    return df

def sumif(df,columns_1,n,columns_2):
    #columns_2 是条件判断之后的1 或者 0
    s=pd.Series(df[f'{columns_1}']*df[f'{columns_2}'].
                rolling(window=n, center = False,
                        min_periods = minp(n)).sum(),
                name=f'sumif_{columns_1}_{columns_2}_' + str(n))
    df = df.join(s)
    return df

def minp(d):
    """
    :param d: Pandas rolling 的 window
    :return: 返回值为 int，对应 window 的 min_periods
    """
    if not isinstance(d, int):
        d = int(d)
    if d <= 10:
        return d - 1
    else:
        return d * 2 // 3

def count(df,columns, n):
    df[f'{columns}'] = df[f'{columns}'].fillna(0)# columns是条件判断之后的1或者0
    c=pd.Series(df[f'{columns}'].rolling(window = n, center = False,
                                         min_periods=minp(n)).sum(),
                name=f'count_{columns}_' + str(n))
    df = df.join(c)
    return df


class factor_1():
    def __init__(self, data_0=pd.DataFrame()):
        """
        @Description : init
        @Params      :
        @Returns     :
        """
        self.data_0 = data_0

    # @profile
    def load_data(self):
        """
        @Description :
        @Params      :
        @Returns     :
        """
        default_start = '20170101'
        default_end = '20221015'

        self.data_0 = pd.read_csv(
            '/Users/irenewang/Desktop/QF603 - Quantitative Analysis/Project/ashareeodprices_total.csv.gz')
        self.data_0.sort_values(by=['ukey', 'DataDate'], inplace=True)
        self.data_0 = self.data_0.astype(float)
        self.data_0['ukey'] = self.data_0['ukey'].astype(int)
        self.data_0['DataDate'] = self.data_0['DataDate'].astype(int)
        self.data_0 = self.data_0.query(
            f'DataDate < {default_end} and DataDate >= {default_start}')

    def generate(self):
        """
        formula: (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))

        @Description :
        @Params      :
        @Returns     :
        """
        # 参数设定
        delta = 1
        corr = 6

        data_0 = self.data_0.copy()
        data_0['Close_Open'] = (data_0['Close'] - data_0['Open']) / (
                    data_0['Open'] + 0.0001)
        data_0['log_volume'] = np.log(data_0['Volume'])

        df_list = []
        i = 1
        group = data_0.groupby('ukey')
        for _, ukeydf in group:
            print(i, end='\r')
            ukeydf.reset_index(drop=True, inplace=True)
            ukeydf = ts_delta(ukeydf, columns='log_volume', n=delta)
            df_list.append(ukeydf)
            i += 1
        data_0 = pd.concat(df_list, axis=0)

        df_list = []
        i = 1
        group = data_0.groupby('DataDate')
        for _, ukeydf in group:
            print(i, end='\r')
            ukeydf.reset_index(drop=True, inplace=True)
            ukeydf['rank_ts_delta'] = ukeydf[f'ts_delta_{delta}'].rank(
                ascending=False)
            ukeydf['rank_Close_Open'] = ukeydf['Close_Open'].rank(
                ascending=False)
            df_list.append(ukeydf)
            i += 1
        data_0 = pd.concat(df_list, axis=0)
        data_0.sort_values(by=['ukey', 'DataDate'], inplace=True)

        df_list = []
        i = 1
        group = data_0.groupby('ukey')
        for _, ukeydf in group:
            print(i, end='\r')
            ukeydf.reset_index(drop=True, inplace=True)
            ukeydf = ts_corr(ukeydf, columns_1='rank_ts_delta',
                             columns_2='rank_Close_Open', n=corr)
            df_list.append(ukeydf)
            i += 1
        data_0 = pd.concat(df_list, axis=0)
        data_0.eval(f"x = - ts_corr_{corr}", inplace=True)
        data_0.drop_duplicates(subset=['ukey', 'DataDate'], inplace=True)

        factor = data_0[['ukey', 'DataDate', 'x']].dropna()
        factor['DataDate'] = factor['DataDate'].astype(str)

        factor.to_csv('factor_alpha_1.csv.gz')


if __name__ == "__main__":
    factor = factor_1()
    factor.load_data()
    factor.generate()
