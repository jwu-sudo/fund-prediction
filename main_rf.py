# -*- coding: utf-8 -*-
# @Time : 2020/11/17 8:45 下午 
# @Author : jwu
# @File : main.py 
# @Description: This is a python script that implements different kinds of machine learning algorithm

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

def select(startdate, enddate, pathsplit="HQ/tick_159919/splitdata.h5"):
    '''
    select dataframe stored in .h5 file under pathspllit
    :param startdate: date the selection begins
    :param enddate: date the selection ends(include)
    :param pathsplit: path store .h5 file
    :return: whole data as a dataframe from startdate to enddate
    '''
    with pd.HDFStore(pathsplit, 'r') as split:
        keys = list(split.keys())
        keys = [i for i in keys if i>=pathsplit[2:14]+"_"+str(startdate) and i<=pathsplit[2:14] + "_" + str(enddate)]
        df = pd.DataFrame()
        for i in keys:
            df = df.append(split[i])
    return df

class MYModel(object):
    def __init__(self, model):
        '''
        template for model class
        :param model: params used in the model
        '''
        self.model = model

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

class RF(MYModel):
    def __init__(self, params):
        '''
        This class implements random forest model
        :param params: params used in rf model
        '''
        self.params = params
        self.model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                            oob_score=params['oob_score'],
                                            max_features=params['max_features'],
                                            max_depth=params['max_depth'],
                                            min_samples_split=params['min_samples_split'],
                                            min_samples_leaf=params['min_samples_leaf'],
                                            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                            max_leaf_nodes=params['max_leaf_nodes'],
                                            min_impurity_decrease=params['min_impurity_decrease'],
                                            n_jobs=params['n_jobs'],
                                            random_state=params['random_state']
                                            )

    def update_params(self, params):
        self.params = params
        self.model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                            oob_score=params['oob_score'],
                                            max_features=params['max_features'],
                                            max_depth=params['max_depth'],
                                            min_samples_split=params['min_samples_split'],
                                            min_samples_leaf=params['min_samples_leaf'],
                                            min_weight_fraction_leaf=params['min_weight_fraction_leaf'],
                                            max_leaf_nodes=params['max_leaf_nodes'],
                                            min_impurity_decrease=params['min_impurity_decrease'],
                                            n_jobs=params['n_jobs'],
                                            random_state=params['random_state']
                                            )

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def gridsearch(self, x_train, y_train):

        # grid search for n_estimators
        param_opt = {'n_estimators': range(60, 100, 10)}
        gsearch = GridSearchCV(estimator=self.model,
                               param_grid=param_opt,
                               cv=3,
                               verbose=1)
        gsearch.fit(x_train, y_train)
        print(gsearch.cv_results_, '\n', gsearch.best_params_, '\n', gsearch.best_score_)
        n_estimators = gsearch.best_params_['n_estimators']
        params = {'n_estimators': n_estimators,
                  'oob_score': True,
                  'max_features': 'auto',
                  'max_depth': 5,
                  'min_samples_split': 100,
                  'min_samples_leaf': 10,
                  'min_weight_fraction_leaf': 0,
                  'max_leaf_nodes': None,  # over fitting
                  'min_impurity_decrease': 0,
                  'n_jobs': None,
                  'random_state': 0
                  }
        self.update_params(params)

        param_opt = {'max_depth': range(3, 12, 1)}
        gsearch = GridSearchCV(estimator=self.model,
                               param_grid=param_opt,
                               cv=3,
                               verbose=2)
        gsearch.fit(x_train, y_train)
        print(gsearch.cv_results_, '\n', gsearch.best_params_, '\n', gsearch.best_score_)
        max_depth = gsearch.best_params_['max_depth']
        params = {'n_estimators': n_estimators,
                  'oob_score': True,
                  'max_features': 'auto',
                  'max_depth': max_depth,
                  'min_samples_split': 100,
                  'min_samples_leaf': 10,
                  'min_weight_fraction_leaf': 0,
                  'max_leaf_nodes': None,  # over fitting
                  'min_impurity_decrease': 0,
                  'n_jobs': None,
                  'random_state': 0
                  }
        self.update_params(params)
        param_opt = {'min_samples_split': range(90,200,20)}
        gsearch = GridSearchCV(estimator=self.model,
                               param_grid=param_opt,
                               cv=3,
                               verbose=2)
        gsearch.fit(x_train, y_train)

        print(gsearch.cv_results_, '\n', gsearch.best_params_, '\n', gsearch.best_score_)
        min_samples_split = gsearch.best_params_['min_samples_split']
        params = {'n_estimators': n_estimators,
                  'oob_score': True,
                  'max_features': 'auto',
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': 10,
                  'min_weight_fraction_leaf': 0,
                  'max_leaf_nodes': None,  # over fitting
                  'min_impurity_decrease': 0,
                  'n_jobs': None,
                  'random_state': 0
                  }
        self.update_params(params)
        return params
    
    def randomsearch(self, x_train, y_train):

        param_dist = {'n_estimators': randint(100, 2000),
                      'oob_score': [True],
                      'max_features': ['auto','sqrt','log2',None],
                      'max_depth': randint(2, 10),
                      'min_samples_split': randint(5, 100),
                      'min_samples_leaf': randint(5, 100),
                      'min_weight_fraction_leaf': [0],
                      'max_leaf_nodes': [None],  # over fitting
                      'min_impurity_decrease': [0],
                      'n_jobs': [1],
                      'random_state': [0, 15]
                      }

        n_iter = 20
        random_search = RandomizedSearchCV(self.model, 
                                           param_distributions=param_dist,
                                           n_iter=n_iter, 
                                           cv=3, 
                                           verbose=2, 
                                           scoring='accuracy')
        random_search.fit(x_train, y_train)
        print(random_search.cv_results_, random_search.best_params_, random_search.best_score_)
        n_estimators = random_search.best_params_['n_estimators']
        max_features = random_search.best_params_['max_features']
        max_depth = random_search.best_params_['max_depth']
        min_samples_split = random_search.best_params_['min_samples_split']
        min_samples_leaf = random_search.best_params_['min_samples_leaf']
        random_state = random_search.best_params_['random_state']

        params = {'n_estimators': n_estimators,
                  'oob_score': True,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'min_weight_fraction_leaf': 0,
                  'max_leaf_nodes': None,  # over fitting
                  'min_impurity_decrease': 0,
                  'n_jobs': None,
                  'random_state': random_state
                  }
        self.update_params(params)
        return params



class Evaluate(object):
    def __init__(self, model, startdate, enddate, n_train=200, n_test=1, pathsplit="HQ/tick_159919/splitdata.h5", pathmat="result/daily_predicts.h5"):
        with pd.HDFStore(pathsplit, 'r') as split:
            keys = split.keys()
            dates = pd.date_range(str(startdate),str(enddate)).strftime("%Y%m%d").to_list()
            #dates = [pathsplit[2:14]+"_"+i for i in dates]
            keys = [i[-8:] for i in keys]
            dates = list(set(keys) & set(dates))
            self.dates = sorted(dates)
            self.model = model
            self.n_train = n_train
            self.n_test = n_test
            self.pathsplit = pathsplit
            self.pathmat = pathmat
            self.result = pd.DataFrame(columns=['date','obs','ratio','mse'])

    def process(self):
        cur_idx = self.n_train
        n = len(self.dates)
        if os.path.exists(self.pathmat):
            print("File " + self.pathmat + ' already exists, do you want to override it?[yes/no]')
            a = input()
            if a=='yes':
                os.remove(self.pathmat)
            else:
                exit(1)
        math5 = pd.HDFStore(self.pathmat, 'w', complevel=8, complib='blosc')
        while cur_idx <= n-self.n_test:
            startdate = self.dates[cur_idx-self.n_train]
            curdate = self.dates[cur_idx]
            enddate = self.dates[cur_idx+self.n_test-1]
            df = select(curdate, enddate, self.pathsplit)
            n_test = df.shape[0]
            df = select(startdate,enddate,self.pathsplit)
            x_train = df.iloc[:-n_test,:-2].values
            y_train = df.iloc[:-n_test,-1].values
            x_test = df.iloc[-n_test:,:-2].values
            y_test = df.iloc[-n_test:,-1].values
            self.model.fit(x_train, y_train)
            y_pred = self.model.predict(x_test)
            mat = pd.DataFrame(index=df.index[-n_test:])
            mat['mid'] = (df['offer_price1'].values[-n_test:] + df['bid_price1'].values[-n_test:])/2
            mat['y_true'] = y_test
            mat['y_pred'] = y_pred
            math5['mat'+str(enddate)] = mat
            mat['date'] = mat.index.date
            result = pd.DataFrame()
            result['obs'] = mat.groupby(['date']).size()
            result['ratio'] = mat.groupby(['date']).apply(lambda di:np.sum(di['y_true']==di['y_pred'])/di.shape[0])
            result['mse'] = mat.groupby(['date']).apply(lambda di:np.sum(np.square(di['y_true']-di['y_pred']))/di.shape[0])
            print(result)
            self.result = self.result.append(result.reset_index())
            self.result.to_csv('result/random_forest_res.csv')

            cur_idx += self.n_test
        math5.close()

    def show(self):
        print(self.result)

if __name__=="__main__":
    pathsplit = "HQ/tick_159919/splitdata.h5"
    params = {'n_estimators': 262,
              'oob_score': True,
              'max_features': 'auto',
              'max_depth': 3,
              'min_samples_split': 128,
              'min_samples_leaf': 78,
              'min_weight_fraction_leaf': 0,
              'max_leaf_nodes': None, # over fitting
              'min_impurity_decrease': 0,
              'n_jobs': None,
              'random_state': 15
              }

    gridsearch = True
    if gridsearch:
        df = select(20180101, 20180601, pathsplit)
        x_train = df.iloc[:, :-2].values
        y_train = df.iloc[:, -1].values
        model = RF(params)
        params = model.gridsearch(x_train,y_train)
        model.update_params(params)
        print("model final params")
        print(model.params)
        model.fit(x_train, y_train)
        y_predprob = model.model.predict_proba(x_train)
        y_pred = model.model.predict(x_train)
        print("Accuracy Score (Train): %f" % accuracy_score(y_train, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_train, y_predprob, multi_class='ovr'))

    randomsearch = False
    if randomsearch:
        df = select(20160101, 20170101, pathsplit)
        x_train = df.iloc[:, :-2].values
        y_train = df.iloc[:, -1].values
        model = RF(params)
        params = model.randomsearch(x_train,y_train)
        model.update_params(params)
        print("model final params")
        print(model.params)
        model.fit(x_train, y_train)
        y_predprob = model.model.predict_proba(x_train)
        y_pred = model.model.predict(x_train)
        print("Accuracy Score (Train): %f" % accuracy_score(y_train, y_pred))
        print("AUC Score (Train): %f" % roc_auc_score(y_train, y_predprob, multi_class='ovr'))

    model = RF(params)
    eva = Evaluate(model, 20160301, 20180530, n_train=200, n_test=1, pathsplit="HQ/tick_159919/splitdata.h5", pathmat="result/daily_predicts_rf.h5")
    eva.process()
    eva.result.to_csv('result/rf_res.csv')