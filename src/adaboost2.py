# -*- coding: utf-8 -*-

import numpy as np
from numpy import *
import pandas as pd
from sklearn.metrics import classification_report

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

train_Y[train_Y != 0.0] = 1.0
train_Y[train_Y == 0.0] = -1.0
test_Y[test_Y != 0.0] = 1.0
test_Y[test_Y == 0.0] = -1.0

#构建一个简单的单层决策树，作为弱分类器
#D作为每个样本的权重，作为最后计算error的时候多项式乘积的作用
#三层循环
#第一层循环，对特征中的每一个特征进行循环，选出单层决策树的划分特征
#对步长进行循环，选出阈值
#对大于，小于进行切换
#特征：dimen，分类的阈值是 threshVal,分类对应的大小值是threshIneq
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))  #numSteps作为迭代这个单层决策树的步长
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();#第i个特征值的最大最小值
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

#基于单层决策树的AdaBoost的训练过程
#numIt 循环次数，表示构造40个单层决策树
def adaBoostTrainDS(dataArr,classLabels,numIt=10):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))  #这里还用到一个sign函数，主要是将概率可以映射到-1,1的类型
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

#train data
weak, agg = adaBoostTrainDS(train_X, train_Y)
print '====='
print weak
print len(weak)
print '====='
print agg
print agg.shape
print '====='
res = adaClassify(test_X, weak)
print res;
print (classification_report(test_Y,res))
