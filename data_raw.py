# -*- coding: utf-8 -*-
# @Time : 2020/11/14 4:36 下午
# @Author : jwu
# @File : data_raw.py
# @Description: This is a python script manipulate and analysis the data

import pandas as pd
import numpy as np
import tqdm
import os

BASE_POINT = 10000

## read txt file and store as csv dataframe
etf = ['159919', '510300']
path_hq = ['HQ/tick_' + i for i in etf]
for path_hqi in path_hq:
    files = os.listdir(path_hqi)
    files = [i for i in files if not (("csv" in i) or (i[0]=='.') or (i[-1]=='5')) ]
    files = sorted(files)
    if os.path.exists(path_hqi+'/rawdata.h5'):
        print("File " + path_hqi+'/rawdata.h5' + ' already exists, do you want to override it?[yes/no]')
        a = input()
        if a=='yes':
            os.remove(path_hqi+'/rawdata.h5')
        else:
            exit(1)
    h5 = pd.HDFStore(path_hqi+'/rawdata.h5', 'w', complevel=8, complib='blosc')
    for fi in range(len(files)):
        print(path_hqi+'/'+files[fi])
        df = pd.read_json(path_hqi+'/'+files[fi], lines=True)
        df = df.dropna(axis=0, subset=[108, 109, 110, 111])
        df = df.reset_index(drop=True)
        #print(df.head())

        orderbook = {108: "offer_price", 109: "offer_volume", 110: "bid_price", 111: "bid_volume"}
        other_info = {100: "preclose", 101: "open", 102: "high", 103: "low", 104: "last",
                      112: "num_trades", 113: "total_volume_trade", 114: "total_value_trade",
                      115: "total_bid_qty", 116: "weighted_avg_bid_price",
                      118: "total_offer_qty", 119: "weighted_avg_offer_price",
                      123: "high_limited_price", 124: "low_limited_price"}
        n = df.shape[0]
        column = list(orderbook.values())
        num = [str(i) for i in range(1,11)]
        column = [i+j for i in column for j in num]

        data = []
        for i in tqdm.tqdm(range(n)):
            ls = []
            for key, value in orderbook.items():
                arr = df.loc[i, key]
                arr += [0]*(10-len(arr))
                ls.extend(arr)
            data.append(ls)
        data = pd.DataFrame(data, columns=column)/BASE_POINT
        data.describe()

        df['datetime'] = df.loc[:,3].astype(str) + df.loc[:,4].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S%f')

        data['datetime'] = df['datetime']
        for key, value in other_info.items():
            data[value] = df[key] if (key in df.columns) else np.nan

        data.loc[:,['preclose', 'open', 'high', 'low','last','weighted_avg_offer_price','weighted_avg_bid_price',]] /= BASE_POINT

        data = data.set_index(['datetime'])
        h5[files[fi][:-4]] = data
        #data.to_csv(path_hqi+'/'+files[fi][:-4]+".csv")
    h5.close()