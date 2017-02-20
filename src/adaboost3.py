import numpy as np
from numpy import *
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import AdaBoostClassifier

TRAIN = 18000
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
#OVR
Pre_Y = OneVsRestClassifier(AdaBoostClassifier(n_estimators=100, learning_rate=0.5)).fit(train_X, train_Y).predict(test_X)
print (classification_report(test_Y,Pre_Y))
