import numpy as np
from numpy import *
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

TRAIN = 16000
filename = "../data/letter-recognition.data"
col_name = ['lettr','x-box','y-box','width','high','onpix',\
'x-bar','y-bar', 'x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']
print len(col_name)
data = pd.read_csv(filename, names = col_name)
data_X = np.array(data.ix[:,'x-box':],dtype='float')
format = lambda X: ord(X) - ord('A')
data_Y = np.array(data['lettr'].map(format),dtype='float')
train_X = data_X[:TRAIN]
train_Y = data_Y[:TRAIN]
test_X = data_X[TRAIN:]
test_Y = data_Y[TRAIN:]
print train_X.shape, train_Y.shape, test_X.shape, test_Y.shape

#Prd_Y =  OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_X, train_Y).predict(test_X)
clf = RandomForestClassifier(n_estimators=100)
clf = clf.fit(train_X, train_Y)
Prd_Y = clf.predict(test_X)
print Prd_Y
print (classification_report(test_Y,Prd_Y))
