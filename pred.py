#!/usr/bin/env python

from __future__ import division

import os, sys, os.path
import random, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import  roc_auc_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
from matplotlib.pylab import rcParams

def readData(_link) :
    print 'reading file: ', _link
    _df = pd.read_csv(_link,header=0)
    #print _df['TARGET'].value_counts()
    return _df

#===============================================================================
plt.figure()
#FIXME
#_path = 'SubSubData'
_path = 'Data'

train = readData(('%s/train.csv'%_path))
test  = readData(('%s/test.csv'%_path))
#features = train.columns.values
features = train.columns

empty = []
for fea in features :
    if train[fea].std() == 0 :
        empty.append(fea)
train = train.drop(empty, axis=1)
test  = test.drop(empty, axis=1)


Xtrain = train.drop(['TARGET','ID'], axis=1)
Ytrain = train['TARGET'].values
testid = test['ID']
test=test.drop('ID',axis=1)


#alg = RandomForestClassifier(random_state=1, n_estimators=150,
#        min_samples_split=8, min_samples_leaf=4,max_depth=4)

#alg = GradientBoostingClassifier(random_state=1, n_estimators=3, max_depth=2)

alg = ExtraTreesClassifier(random_state=1)

Xcv_train, Xcv_test, Ycv_train, Ycv_test = train_test_split(Xtrain, Ytrain,
        test_size=0.2, random_state=1)
Ycv_test = Ycv_test.astype(float)
selector = alg.fit(Xcv_train,Ycv_train)
features_by_imp = pd.Series(alg.feature_importances_,
        index=Xcv_train.columns.values).sort_values(ascending=False)
features_by_imp[:30].plot(kind='bar',title='Most imp. features', figsize=(12,8))
#plt.show()
plt.savefig('best_features.png',format='PNG')


#imp_features = []
#for fea,imp in features_by_imp :
#    print fea, '::::',imp

sfm = SelectFromModel(selector, prefit=True)
Xcv_train = sfm.transform(Xcv_train)
Xcv_test  = sfm.transform(Xcv_test)
test = sfm.transform(test)
print ('new shape: ', Xcv_train.shape, Xcv_test.shape)

#scores = cross_validation.cross_val_score(alg,Xtrain,Ytrain,cv=3)
#print scores

print 'Ycv_test: ', Ycv_test
m2_xgb = xgb.XGBClassifier(base_score=0.5,n_estimators=110, nthread=-1, max_depth = 4, seed=1)
m2_xgb.fit(Xcv_train, Ycv_train, eval_metric="auc", verbose = False,
                   eval_set=[(Xcv_test, Ycv_test)])
print("Roc AUC: ", roc_auc_score(Ycv_test, m2_xgb.predict_proba(Xcv_test)[:,1],
                          average='macro'))


probs = m2_xgb.predict_proba(test)


##-------------------------------------------------------------------------------
#alg.fit(Xtrain,Ytrain)
#
#probs = alg.predict_proba(test).astype(float)

#---
submission = pd.DataFrame({'ID':testid, 'TARGET':probs[:,1]})
submission.to_csv('submission.csv',index=False)

#===============================================================================
#for fea in features :
#    plt.plot(train[fea])
#    plt.savefig('plots/%s.png' % fea)
#    plt.clf()

#predictions = alg.predict(test) #.astype(int)
#print predictions
#submission = pd.DataFrame({'ID':test['ID'], 'TARGET':predictions})
#submission.to_csv('submission.csv',index=False)

