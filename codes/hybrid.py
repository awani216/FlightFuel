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
@jit
def func(dataset, labels):
    clf_arr = [tree.DecisionTreeRegressor(), RandomForestRegressor(), 
    KNeighborsRegressor(), ExtraTreesRegressor()]
    clf_nm = [ "Decision Tree Regressor", "Random Forest Regressor", 
    "K Neighbors Regressor", "Extra Trees Regressor"]
    test_dataset = dataset[-15000:]
    test_labels = labels[-15000:]
    dataset = dataset[:-15000]
    labels = labels[:-15000]    
    result = np.empty(shape=(len(test_dataset), len(clf_arr)))
    acc = []
    sum_acc = 0
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
        sum_acc += acc[i]*acc[i]
    ct = 0
    res_wa = []
    res_mid = []
    res_mindiff = []


    # Weighted mean of samples (weights = accuracy of the classifier)
    for i in range(len(result)):
        temp = 0        
        for j in range(len(clf_arr)):
            temp += result[i,j]*acc[j]*acc[j]
        res_wa.append(temp/sum_acc)
        if abs(res_wa[i] - test_labels[i]) <= 120:
            ct += 1
    print("Final accuracy by Wighted Mean Method is", 100*ct/len(test_dataset))
    ct = 0
    # Taking Median value
    for i in range(len(result)):
        temp = []
        for j in range(len(clf_arr)):
            temp.append(result[i,j])
        temp.sort()
        res_mid.append((temp[1] + temp[2])/2)
        if abs(res_mid[i] - test_labels[i]) <= 120:
            ct += 1
    print("Final accuracy by Median Method is", 100*ct/len(test_dataset)) 
    ct = 0
    for i in range(len(result)):
        temp = []
        for j in range(len(clf_arr)-2):
            temp.append([result[i,j+2] - result[i,j], j])
        temp.sort()
        temp = np.array(temp, dtype=int)
        z = temp[0,1]
        res_mindiff.append(result[i,z+1])
        if abs(res_mindiff[i] - test_labels[i]) <= 120:
            ct += 1
    print("Final accuracy by Minimum Difference Method is", 100*ct/len(test_dataset))

    acc_1 = []
    for j in range(len(acc)):
        acc_1.append([acc[j], j])
    acc_1.sort()
    acc_1 = np.array(acc_1)
    max1 = int(acc_1[0,1])
    max2 = int(acc_1[1,1])
    ct = 0
    s = 0
    for i in range(len(result)):
        res = result[i, max1] + 0.1*result[i, max2] + 20
        s += result[i,max1] - test_labels[i]
        res = res/1.1
        if abs(res - test_labels[i]) <= 120:
            ct += 1
    print("Final accuracy by Hybrid is", 100*ct/len(test_dataset))
    print(s)



func(train_dataset, train_labels)