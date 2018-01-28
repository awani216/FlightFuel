import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from six.moves import cPickle as pickle
import random


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
@jit
def func():    
    
    clf = KNeighborsClassifier()
    ftime = time.time()
    clf.fit(train_dataset[:50000], train_labels[:50000])
    ftime = time.time()
    x_test = train_dataset[-1000:]
    y_test = train_labels[-1000:]
    res = clf.predict(x_test)
    ct = 0
    for i in range(len(res)):
        if abs(res[i] - y_test[i]) <= 200:
            ct += 1
    print("Time Taken ", time.time()-ftime)
    print("Accuracy in step " + str(100*(ct/len(res))))
    

func()
