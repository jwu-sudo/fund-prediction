# -*- coding: utf-8 -*-
# @Time : 2020/11/17 8:46 下午 
# @Author : jwu
# @File : data_split.py 
# @Description: This is a python script to store the data suitable for ML model

import pandas as pd
import numpy as np
import os

BASE_POINT = 10000

def raw2xgboost(freq, pathdata="HQ/tick_510300/rawdata.h5", pathsplit="HQ/tick_510300/splitdata.h5"):
    if os.path.exists(pathsplit):
        print("File " + pathsplit + ' already exists, do you want to override it?[yes/no]')
        a = input()
        if a == 'yes':
            os.remove(pathsplit)
        else:
            exit(1)
    with pd.HDFStore(pathsplit, 'w') as split:
        with pd.HDFStore(pathdata, 'r') as raw:
            keys = raw.keys()
            print(keys)
            for key in keys:
                df = raw[key]
                df = df.loc[:,['offer_price1', 'offer_price2', 'offer_price3', 'offer_price4',
                               'offer_price5', 'offer_price6', 'offer_price7', 'offer_price8',
                               'offer_price9', 'offer_price10', 'offer_volume1', 'offer_volume2',
                               'offer_volume3', 'offer_volume4', 'offer_volume5', 'offer_volume6',
                               'offer_volume7', 'offer_volume8', 'offer_volume9', 'offer_volume10',
                               'bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5',
                               'bid_price6', 'bid_price7', 'bid_price8', 'bid_price9', 'bid_price10',
                               'bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4',
                               'bid_volume5', 'bid_volume6', 'bid_volume7', 'bid_volume8',
                               'bid_volume9', 'bid_volume10',
                               'preclose', 'open', 'high', 'low', 'last',
                               "num_trades", "total_volume_trade", "total_value_trade",
                               'total_bid_qty', 'weighted_avg_offer_price',
                               'total_offer_qty', 'weighted_avg_bid_price']]

                df = df.loc[(df.index > pd.to_datetime(key[-8:]+" 09:15:00")) & (
                            df.index < pd.to_datetime(key[-8:]+" 15:00:00"))]

                df = df.sort_index()
                #df = df.drop_duplicates()
                df['keep'] = 1
                for i in range(df.shape[0] - 1):
                    if np.all(df.iloc[i, :] == df.iloc[i + 1, :]):
                        if (df.iloc[i, :]['num_trades'] == 0) or (df.index[i] == df.index[i + 1]):
                            df.iloc[i + 1, -1] = 0
                    if df.index[i] == df.index[i + 1]:
                        df.loc[df.index[i],'num_trades'] += df.iloc[i + 1, :]['num_trades']
                        df.iloc[i + 1, -1] = 0
                df = df.loc[df['keep'] == 1]
                df = df.drop(['keep'], 1)

                target = pd.DataFrame(index=df.index, columns=['nexttime','mid','nextmid'])
                idx = sorted(list(df.index))
                idxgap = [i + pd.Timedelta(str(freq)+'s') for i in idx]
                newidx = []
                i = 0
                j = 0
                while i < len(idxgap) and j < len(idx):
                    if idxgap[i] > idx[j]:
                        j += 1
                    else:
                        i += 1
                        newidx.append(idx[j])
                        target.loc[idx[i], :] = np.array(
                            [idx[j], (df.loc[idx[i], 'offer_price1'] + df.loc[idx[i], 'bid_price1']) / 2,
                             (df.loc[idx[j], 'offer_price1'] + df.loc[idx[j], 'bid_price1']) / 2])
                df['ret'] = target['nextmid'] / target['mid'] - 1
                df = df.replace(np.inf, np.nan)
                df['sign'] = (df['ret'] > 0).astype(int)
                df.loc[df['ret'] < 0, 'sign'] = -1
                df = df.dropna(0, how='any')
                split[key] = df
                print(df.shape)
                print(df.head())

if __name__=='__main__':
    #raw2xgboost(10, "HQ/tick_159919/rawdata.h5", "HQ/tick_159919/splitdata.h5")
    raw2xgboost(10, "HQ/tick_510300/rawdata.h5", "HQ/tick_510300/splitdata10s.h5")