# -*- coding: utf-8 -*-
# @Time : 2020/11/20 3:36 下午 
# @Author : jwu
# @File : EDA.py 
# @Description: This is a python script explore the data and do sample analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

def select(startdate, enddate, pathsplit="HQ/tick_159919/splitdata.h5"):
    with pd.HDFStore(pathsplit, 'r') as split:
        keys = list(split.keys())
        keys = [i for i in keys if i>=pathsplit[2:14]+"_"+str(startdate) and i<=pathsplit[2:14] + "_" + str(enddate)]
        df = pd.DataFrame()
        for i in keys:
            df = df.append(split[i])
    return df

def eda(df):
    print(df.info())
    desc = df.describe()
    index = desc.index[1:]
    desc = desc.loc[index]
    columns = desc.columns
    colors = ['blue', 'red', 'green', 'black', 'pink', 'purple', 'gray', 'yellow']
    for i in range(6):
        plt.figure(figsize=(16, 16))
        for ii in range(9):
            ax = plt.subplot(3, 3, ii + 1)
            ax.set_title(columns[i * 9 + ii])
            for j in range(len(index)):
                plt.bar(index[j], desc.loc[index[j], columns[i * 9 + ii]], color=colors[j])
        plt.savefig("picture/stat" + columns[i * 9 + ii])
    df['mid'] = (df['offer_price1'] + df['bid_price1']) / 2
    df.loc[df['last'] != 0, ['last', 'mid']].plot(figsize=(16, 8))

    corr = df.loc[:, ['bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5',
                      'bid_price6', 'bid_price7', 'bid_price8', 'bid_price9', 'bid_price10']].corr()
    plt.figure(figsize=(16, 16))
    plt.title("Correlation of features")
    sns.heatmap(corr, square=True)
    plt.savefig("picture/corr_bidprice")

    corr = df.loc[:, ['bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4',
                      'bid_volume5', 'bid_volume6', 'bid_volume7', 'bid_volume8',
                      'bid_volume9', 'bid_volume10']].corr()
    plt.figure(figsize=(16, 16))
    plt.title("Correlation of features")
    sns.heatmap(corr, square=True)
    plt.savefig("picture/corr_bidvolume")

    corr = df.loc[:, ['offer_volume1', 'offer_volume2','offer_volume3', 'offer_volume4', 'offer_volume5',
                      'offer_volume6', 'offer_volume7', 'offer_volume8', 'offer_volume9', 'offer_volume10']]
    low = 0.1
    high = 0.9
    quant_df = corr.quantile([low, high])
    filt_df = corr.apply(lambda x: x[(x > quant_df.loc[low, x.name]) &
                                     (x < quant_df.loc[high, x.name])], axis=0)
    plt.figure(figsize=(16, 16))
    plt.title("Correlation of features")
    sns.heatmap(filt_df.corr(), square=True)
    plt.savefig("picture/corr_offervolume_new")

    for i in range(df.shape[1]):
        sns.displot(df, x=df.columns[i], rug=True)
        plt.show()

def feature_importance(x_train, y_train, columns):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(x_train, y_train)
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(x_train.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, f + 1, columns[indices[f]], importances[indices[f]]))
    # Plot the feature importances
    plt.figure(figsize=(16, 8))
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), indices)
    plt.xlim([-1, x_train.shape[1]])
    plt.show()

if __name__=="__main__":
    df = select(20160104, 20170101)
    eda(df)
    x_train = df.iloc[:, :-2]
    columns = x_train.columns
    x_train = x_train.values
    y_train = df.iloc[:, -1].values
    feature_importance(x_train,y_train,columns)