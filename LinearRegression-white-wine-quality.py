5# -*- coding: utf-8 -*-

"""
Linear Regression Classifier - White wine quality

"""
from sklearn.linear_model import LinearRegression

def feature(featureDatafile, splitCh =','):
    infile=open(featureDatafile,'r')
    
    feature={}
    ID=1
    for line in infile:
        feature[ID]=line[:-1].split(splitCh)
        ID+=1
    infile.close()
    return feature

wwq=feature('winequality-white.txt',';')

#Convert wwq entries to numeric
for key in wwq:
    wwq[key]=[float(a) for a in wwq[key]]
    
X,Y = [],[]
for key in wwq:
    X.append(wwq[key][:-1])
    Y.append(wwq[key][-1])

wineReg = LinearRegression()
wineReg.fit(X,Y)

def predictor(coef,intercept,test):
    """ Returns the prediction for Y in a linear regression where coef is the list of coefficents, intercept is the intercept, test is an unclassified vector"""
    
    y=0
    for i in range(len(coef)):
        y = y + coef[i]*test[i]
    y = y + intercept
    return y

#Testing the model. Training set and a test set .

from random import sample

sampleKeys = sample(wwq.keys(),2400)
train = {}
test = {}

for key in wwq:
    if key in sampleKeys:
        test[key]=wwq[key]
    else:
        train[key]=wwq[key]
    
#Use train dictionary to form a linear regression model

X,Y = [],[]
for key in train:
    X.append(train[key][:-1])
    Y.append(train[key][-1])

trainReg = LinearRegression()
trainReg.fit(X,Y)

#Evaluate the model on test data

def evaluate(LRmodel, testDict,error = 0.5):
    correct,incorrect = 0,0
    
    for key in testDict:
        p = predictor(LRmodel.coef_,LRmodel.intercept_,testDict[key][:-1])
        if abs(p-testDict[key][-1])< error:
            correct+=1
        else:
            incorrect+=1
    return correct, incorrect

print(evaluate(trainReg,test,error=1))

    
    





