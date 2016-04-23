#!/usr/bin/python 

import os, sys, os.path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import svm

plt.figure()

def get_title(name) :
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search :
        return title_search.group(1)
    return ''


def getCabin(row) :
    if not pd.isnull(row) :
        return row[0]
    #else :
    #    print 'NNNNAAAAANNN'
    return ''

family_id_mapping = {}
def get_family_id(row) :
    last_name = row['Name'].split(',')[0]
    family_id = '{0}{1}'.format(last_name,row['FamilySize'])

    if family_id not in family_id_mapping :
        if len(family_id_mapping) == 0 :
            current_id = 1
        else :
            current_id = (max(family_id_mapping.items(),
                        key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id

    return family_id_mapping[family_id]

def buildFeatures(_df) :
    titles = _df['Name'].apply(get_title)
    titlesuni = titles.unique()
    title_mapping =  {t:n for t,n in zip(titlesuni, range(1,len(titlesuni)+1))}

    for k,v in title_mapping.items() :
        titles[titles == k] = v



    _df['Cabin'] = _df['Cabin'].fillna('X')
#    cabins = _df.apply(getCabin, axis=1)
#    cabins = _df['Cabin'].apply(lambda x : x[0])
    
    cabins = _df['Cabin'].apply(getCabin)
    print cabins.value_counts()
    cabuni = cabins.unique()
    cabmap = {c:v for c,v in zip(cabuni,range(len(cabuni)))}
    print cabmap
    for c,v in cabmap.items() :
        cabins[cabins == c] = v
    print cabins

#---
    _df['Title'] = titles
    _df['CAB'] = cabins
    _df['FamilySize'] = _df['SibSp'] + _df['Parch']
    _df["PSA"] = _df["Pclass"]*_df["Sex"]*_df["Age"]
    _df["SP"] = _df["SibSp"]+_df["Parch"]


    # !!! after FamilySize
    family_ids = _df.apply(get_family_id, axis=1)

    #FIXME
    family_ids[_df['FamilySize'] < 3] = -1
    #FIXME

    _df['FamilyId'] = family_ids

    _df['NameLength'] = _df['Name'].apply(lambda x: len(x)+0.0001).astype(np.float)

    return _df

titanic = pd.read_csv('../train.csv',header=0)
test_df = pd.read_csv('../test.csv' ,header=0)

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())


test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())

titanic.loc[titanic['Sex'] == 'male'  , 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
test_df.loc[test_df['Sex'] == 'male'  , 'Sex'] = 0
test_df.loc[test_df['Sex'] == 'female', 'Sex'] = 1


#print titanic.Embarked.unique()
titanic.Embarked = titanic.Embarked.fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

test_df.Embarked = test_df.Embarked.fillna('S')
test_df.loc[test_df['Embarked'] == 'S', 'Embarked'] = 0
test_df.loc[test_df['Embarked'] == 'C', 'Embarked'] = 1
test_df.loc[test_df['Embarked'] == 'Q', 'Embarked'] = 2

#==================================================
#--------------------------------------------------
titanic = buildFeatures(titanic)
test_df = buildFeatures(test_df)

#plt.hist2d(titanic['Survived'],titanic.NameLength)
#plt.show()


predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'FamilySize',  'Fare',
           'Embarked', 'NameLength', 'Title','PSA','SP', 'FamilyId', 'CAB']

#===================================================
# -- select best features --------------------------
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic['Survived'])
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
#plt.show()
plt.savefig('best_features.png',format='PNG')



pred_1 = ['Pclass', 'Sex', 'Fare', 'NameLength', 'Title','PSA'] 
pred_2 = ['Pclass', 'Sex', 'Fare', 'NameLength', 'Title','PSA'] 
pred_3 = ['Pclass', 'Sex', 'Fare', 'NameLength', 'Title','PSA'] 
pred_4 = ['Pclass', 'Sex', 'Fare', 'NameLength', 'Title','PSA'] 
#pred_3 = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'FamilySize',  'Fare', 'Embarked', 'NameLength', 'Title','PSA'] 

algo1 = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
algo2 = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=8, min_samples_leaf=4)
algo3 = LogisticRegression(random_state=1)
algo4 = AdaBoostClassifier(n_estimators=500)
algorithms = [ [algo1,pred_1], [algo2,pred_2], [algo3,pred_3], [algo4,pred_4]]
print 'Len Algo: ', len(algorithms)
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []


wgt = [2.,3.,1.,2.]

for train, test in kf :
    train_target = titanic['Survived'].iloc[train]
    full_test_predictions = []
    for alg, predictors in algorithms :
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    fp = full_test_predictions
    test_predictions = sum([f*w for f,w in zip(fp,wgt)]) / sum(wgt)

    test_predictions[test_predictions <= 0.5] = 0
    test_predictions[test_predictions >  0.5] = 1
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
accuracy = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
print 'accuracy = ', accuracy



full_predictions = []
for alg, predictors in algorithms :
    alg.fit(titanic[predictors], titanic['Survived'])

    predictions = alg.predict_proba(test_df[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
fp = full_predictions
predictions = sum([f*w for f,w in zip(fp,wgt)]) / sum(wgt)
predictions[predictions <= 0.5] = 0
predictions[predictions >  0.5] = 1
predictions = predictions.astype(int)


#######################
#   FOR SUBMISSION
#######################
#alg.fit(titanic[predictors], titanic['Survived'])
#predictions = alg.predict(test_df[predictors])
submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived':predictions})
submission.to_csv('sub_mix.csv',index=False)


##############################################################
