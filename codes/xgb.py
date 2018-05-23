import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import tree
from six.moves import cPickle as pickle
import random
import os, shutil
import xgboost as xgb


#@jit
#def func():
with open(r"../random_forest.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']
train_labels = train_labels/100
train_labels = np.array(train_labels, dtype=int)
#mx = train_labels.max()
#train_labels = train_labels/mx
#err = 200/mx
ran = np.random.permutation(len(train_dataset))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]
mean = np.mean(train_dataset, axis=0)
std = np.std(train_dataset, axis=0)
train_dataset = np.array((train_dataset-mean)/std)
print(train_labels)
test_dataset = train_dataset[-10000:]
test_labels = train_labels[-10000:]
train_dataset = train_dataset[:50000]
train_labels = train_labels[:50000]
dtrain = xgb.DMatrix(train_dataset, label=train_labels)
dtest = xgb.DMatrix(test_dataset, label=test_labels)
print(train_labels.min())
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 2
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 130
plst = param.items()
evallist = [(dtrain, 'train')]
num_round = 100
bst = xgb.train(plst, dtrain, num_round, evallist)
ypred = bst.predict(dtest)
ct=0
for i in range(len(ypred)):
    if (abs(test_labels[i] - ypred[i]) <= 2):
        ct+= 1

print(ct/len(ypred))
