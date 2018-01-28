import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree
import random
import math as m

# Opening data
with open(r"../random_forest.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

# Type casting 
train_labels = np.array(train_labels, dtype=int)

# Scaling by finding mean and variance
mean = np.mean(train_dataset, axis=0)
std = np.std(train_dataset, axis=0)
train_dataset = (train_dataset - mean)/std

ip = []
for i in range(len(attributes)):
    print("Enter value of attribute :", attributes[i],  ", Else press return")
    temp = input()
    if temp is not '':
        ip.append([((float(temp)-float(mean[i]))/std[i]),i])
ip = np.array(ip, dtype=float)
ip_n = []
if len(ip) is not 0:
    ip_n = ip[:1]
ip_n = np.array(ip_n, dtype=int)

# Calculating eucledean distance of array
dist = []
limt = m.sqrt(0.02*len(ip))
arr = []
for i in range(len(train_dataset)):
    a = 0
    for j in range(len(ip)):
        a += abs(train_dataset[i, int(ip[j,1])] - ip[j,0])**2
    a = m.sqrt(a)
    if a <= limt :
        arr.append(i)
    dist.append([a,i])
if len(arr) <= 5:
    arr = []
    limt = 2*limt
    for i in range(len(train_dataset)):
        a = 0
        for j in range(len(ip)):
            a += abs(train_dataset[i, int(ip[j,1])] - ip[j,0])**2
        a = m.sqrt(a)
        if a <= limt :
            arr.append(i)
        dist.append([a,i])

dist.sort(key=lambda x : x[0])

order = []
for i in range(len(arr)):
    order.append([train_labels[arr[i]], i])



order.sort(key=lambda x : x[0])
order = np.array(order)
k = order[:,1]
k = np.array(k, dtype=int)
near_dataset = train_dataset[k]
near_labels = train_labels[k]

# finding  correlation between samples and Fuel Flow
corr = []
arr_corr = []
arr_rf = []
for i in range(len(attributes)):
    sample = near_dataset[:, i]
    corr.append(np.corrcoef(sample, near_labels)[0,1])

print("\nImportant Variables from correlation index:")
for i in range(len(corr)):
    if abs(corr[i]) >= 0.3:
        if i not in ip_n:
            print(attributes[i], corr[i])
            arr_corr.append(i) 
if len(near_dataset) > 1000:
    ran = range(0,1000)
else:
    ran = range(0,len(near_dataset))

# Finding Random Forest importance 

clf = RandomForestClassifier()
clf.fit(near_dataset[ran], near_labels[ran])
imp = clf.feature_importances_
print("\nImportant Variables from random forest:")
for i in range(len(imp)):
    if abs(imp[i]) >= 0.05:
        if i not in ip_n:
            print(attributes[i], imp[i])
            arr_rf.append(i)
arr_corr =  np.array(arr_corr, dtype=int)
arr_rf = np.array(arr_rf, dtype=int)

if len(arr) > 200:
    n_fin = 100
else:
    n_fin = (len(arr)//2) + 1
ans_corr = []
ans_rf = []
clf = RandomForestClassifier()
for i in range(len(arr_corr)):
    ans_corr.append(np.sum(near_dataset[:n_fin, arr_corr[i]])/n_fin)
    
for i in range(len(arr_rf)):
    ans_rf.append(np.sum(near_dataset[:n_fin, arr_rf[i]])/n_fin)


clf = RandomForestClassifier()
ran = random.sample(range(len(train_dataset)), 50000)
clf.fit(train_dataset[ran], train_labels[ran])

print("fitted")
path = r"..\var_rel\test2\plot"
for i in range(len(arr_corr)):
    ran = random.sample(range(len(near_dataset)), 4)
    for j in range(4):        
        dataset = np.tile(near_dataset[j], (100,1))
        maxv = np.amax(near_dataset[:, arr_corr[i]])
        minv = np.amin(near_dataset[:, arr_corr[i]])
        step_size =  (maxv - minv)/100
        pts = np.arange(minv, maxv, step_size)
        dataset[:, arr_corr[i]] = pts[:100]
        ff = clf.predict(dataset)
        ff1 = []
        pts1 = []
        ff2 = []
        pts2 = []
        for k in range(len(ff)):
            if pts[k] < ans_corr[i]:
                pts2.append(pts[k])
                ff2.append(ff[k])
            else:
                pts1.append(pts[k])
                ff1.append(ff[k])
        if corr[arr_corr[i]] < 0:
            temp = ff1
            ff1 = ff2
            ff2 = temp
            temp = pts1
            pts1 = pts2
            pts2 = temp
        plt.clf()
        plt.plot(pts2, ff2, 'go')
        plt.plot(pts1, ff1, 'ro')
        plt.savefig(path + "_corr_" + attributes[arr_corr[i]]  +"_" + str(j) + ".png")
for i in range(len(arr_rf)):
    ran = random.sample(range(len(near_dataset)), 4)
    print(1)
    for j in range(4):        
        dataset = np.tile(near_dataset[j], (100,1))
        maxv = np.amax(near_dataset[:, arr_rf[i]])
        minv = np.amin(near_dataset[:, arr_rf[i]])
        step_size =  (maxv - minv)/100
        pts = np.arange(minv, maxv, step_size)
        dataset[:, arr_rf[i]] = pts[:100]
        ff = clf.predict(dataset)
        ff1 = []
        pts1 = []
        ff2 = []
        pts2 = []
        print(2)
        for k in range(len(ff)):
            if pts[k] < ans_rf[i]:
                pts2.append(pts[k])
                ff2.append(ff[k])
            else:
                pts1.append(pts[k])
                ff1.append(ff[k])
        if corr[arr_rf[i]] < 0:
            temp = ff1
            ff1 = ff2
            ff2 = temp
            temp = pts1
            pts1 = pts2
            pts2 = temp
        print(3)
        plt.clf()
        plt.plot(pts2, ff2, 'go')
        plt.plot(pts1, ff1, 'ro')
        plt.savefig(path + "_rf_" + attributes[arr_rf[i]]  +"_" + str(j) + ".png")



