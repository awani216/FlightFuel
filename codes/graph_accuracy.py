import numpy as np
from numba import jit
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



#@jit
#def func():
with open(r"../random_forest.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

train_labels = np.array(train_labels, dtype=int)
ran = np.random.permutation(len(train_dataset))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]
mean = np.mean(train_dataset, axis=0)
std = np.std(train_dataset, axis=0)
train_dataset = (train_dataset-mean)/std
clf_arr = [KNeighborsRegressor(), tree.DecisionTreeRegressor(), RandomForestRegressor(), 
ExtraTreesRegressor()]
clf_nm = [ "K Neighbors Regressor", "Decision Tree Regressor", "Random Forest Regressor", 
"Extra Trees Regressor"]
dataset = train_dataset
labels = train_labels
test_dataset = dataset[-1000:]
test_labels = labels[-1000:]
dataset = dataset[:-1000]
labels = labels[:-1000]    
result = np.empty(shape=(len(test_dataset), len(clf_arr)))
acc = []
for i in range(len(clf_arr)):
    stime = time.time()
    result_t = np.empty(shape=(len(test_dataset), 7))
    for j in range(7): 
        clf = clf_arr[i]
        ran = random.sample(range(len(dataset)), 50000)
        train_dataset = dataset[ran]
        train_labels = labels[ran]
        clf.fit(train_dataset, train_labels)
        res = clf.predict(test_dataset)
        result_t[:, j] = res
    for j in range(len(test_dataset)):
        temp = result_t[j]
        temp.sort()
        result[j,i] = temp[3]
    
    ct = 0
    for j in range(len(test_dataset)):
        if abs(result[j,i] - test_labels[j]) <= 120:
            ct += 1
    etime = time.time() - stime
    print("Accuracy in ", clf_nm[i], " is ", 100*ct/len(test_dataset), "in time ", etime )
    acc.append(100*ct/len(test_dataset))


with open(r"../data_reduction_4.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

train_labels = np.array(train_labels, dtype=int)
ran = np.random.permutation(len(train_dataset))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]
mean = np.mean(train_dataset, axis=0)
std = np.std(train_dataset, axis=0)
train_dataset = (train_dataset-mean)/std
clf_arr = [KNeighborsRegressor(), tree.DecisionTreeRegressor(), RandomForestRegressor(), 
ExtraTreesRegressor()]
clf_nm = [ "K Neighbors Regressor", "Decision Tree Regressor", "Random Forest Regressor", 
 "Extra Trees Regressor"]
dataset = train_dataset
labels = train_labels
test_dataset = dataset[-1000:]
test_labels = labels[-1000:]
dataset = dataset[:-1000]
labels = labels[:-1000]    
result = np.empty(shape=(len(test_dataset), len(clf_arr)))
acc1 = []
for i in range(len(clf_arr)):
    stime = time.time()
    result_t = np.empty(shape=(len(test_dataset), 7))
    for j in range(7): 
        clf = clf_arr[i]
        ran = random.sample(range(len(dataset)), 50000)
        train_dataset = dataset[ran]
        train_labels = labels[ran]
        clf.fit(train_dataset, train_labels)
        res = clf.predict(test_dataset)
        result_t[:, j] = res
    for j in range(len(test_dataset)):
        temp = result_t[j]
        temp.sort()
        result[j,i] = temp[3]
    
    ct = 0
    for j in range(len(test_dataset)):
        if abs(result[j,i] - test_labels[j]) <= 120:
            ct += 1
    etime = time.time() - stime
    print("Accuracy in ", clf_nm[i], " is ", 100*ct/len(test_dataset), "in time ", etime )
    acc1.append(100*ct/len(test_dataset))
print(acc)
print(acc1)
ind = np.arange(4)
width = 0.35
p1 = plt.bar(ind, acc1, width)
p2 = plt.bar(ind + width, acc, width)
plt.ylabel('Accuracy')
plt.title('Accuracy Based on Models and datasets used')
plt.xticks(ind + width/2, ( "K Neighbors", "Decision Tree", "Random Forest", 
 "Extra Trees"))
plt.legend((p1[0], p2[0]), ('random_forest_importance', 'correlation'))
plt.show()

#func()

