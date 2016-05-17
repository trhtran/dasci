#!/usr/bin/env python

from datetime import datetime
import calendar
import csv

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score

#_dir='SubSubData'
#_dir='SubData'
_dir='Data'
train = pd.read_csv('%s/train.csv'%_dir)
test  = pd.read_csv('%s/test.csv' %_dir)


print '--- digitize week day ----'
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daymap={d:v for d,v in zip(days,range(len(days)+1))}
train_day = train['DayOfWeek']
test_day = test['DayOfWeek']
for d,v in daymap.items():
    train_day[train_day == d] = float(v)
    test_day [ test_day == d] = float(v)
train['WeekDay'] = train_day
test ['WeekDay'] =  test_day

print '--- digitize hour ---'
def getHour(_d) :
    d = datetime.strptime(_d, '%Y-%m-%d %H:%M:%S')
    return d.hour + d.minute/60. + d.second/3600.
train['Hour'] = train['Dates'].apply(lambda x: getHour(x))
test ['Hour'] = test ['Dates'].apply(lambda x: getHour(x))

print '--- digitize day of year ---'
def getDayOfYear(_d) :
    d = datetime.strptime(_d, '%Y-%m-%d %H:%M:%S')
    nbDaysOfMonth = calendar.monthrange(2015,2)[1]
    return d.month + d.day/float(nbDaysOfMonth)
train['DayOfYear'] = train['Dates'].map(lambda x: getDayOfYear(x))
test ['DayOfYear'] = test ['Dates'].map(lambda x: getDayOfYear(x))

ids = test['Id'].values
train = train.drop(['Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'Descript', 'Resolution'],axis=1)
test =test .drop(['Id', 'Dates','DayOfWeek', 'PdDistrict', 'Address'],axis=1)
trainData = train.values
testData  =  test.values

print 'TRAIN:\n',train[:2]
print 'TEST:\n',test [:2]

algo = RandomForestClassifier(random_state = 1, n_estimators = 200,
        min_samples_split=4,min_samples_leaf = 2)

X = trainData[0::,1::]
y = trainData[0::,0]

#print 'cross validating'
#scores = cross_val_score(algo, X, y, cv=3, n_jobs=3)
#print 'scores: {} \n  --> {} +- {}'.format(scores, scores.mean(), scores.std())

print 'fitting'
algo.fit(X,y)


output = algo.predict_proba(testData).astype(float)
print '--- OUTPUT ---\n',output,'\n','-'*20,'\n'

import gzip
predictions_file = gzip.open("submissionRF.csv.gz", "wb")
ofile = csv.writer(predictions_file)
ofile.writerow(["Id",'ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT',
        'DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION',
        'FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT',
        'LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES',
        'PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY',
        'SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY',
        'SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS',
        'WEAPON LAWS'])

for io in range(len(output)):
    ofile.writerow( [io] + list(output[io]) )
predictions_file.close()

##################################################


