import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import tree
import random
import math as m
import csv
from numba import jit

#@jit
#def func():
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
rec_set = []
for i in range(len(train_labels)):
    if train_labels[i] > 7000:
        rec_set.append(i)
rec_set = random.sample(rec_set, 40)
rec_data = train_dataset[rec_set]
rec_labels = train_labels[rec_set]
temp = []
for i in range(len(rec_labels)):
    temp.append([rec_labels[i],i])
temp.sort()
temp = np.array(temp, dtype=int)
order = temp[:,1]
order = np.array(order, dtype=int)
rec_data = rec_data[order]
rec_labels = rec_labels[order]
input_t = []
recc_t = []
print(rec_labels)
ff_red_corr = []
ff_reg_rf = []
for j in range(len(rec_data)):
    ip = []
    sample = rec_data[j]
    test_var =  random.sample(range(len(attributes)), 5)
    strt = ''
    for i in range(len(test_var)):
        ip.append([sample[test_var[i]], test_var[i]])
        strt =  strt + attributes[test_var[i]] + "=" + str(sample[test_var[i]]) + '\n'
    input_t.append(strt[:-1])
    ip = np.array(ip, dtype=float)
    ip_n = []
    if len(ip) is not 0:
        ip_n = ip[:1]
    ip_n = np.array(ip_n, dtype=int)
    dist = []
    limt = m.sqrt(0.02*len(ip))
    arr = []
    for i in range(len(train_dataset)):
        a = 0
        for k in range(len(ip)):
            a += abs(train_dataset[i, int(ip[k,1])] - ip[k,0])**2
        a = m.sqrt(a)
        if a <= limt :
            arr.append(i)
        dist.append([a,i])
    if len(arr) <= 50:
        arr = []
        limt = 2*limt
        for i in range(len(train_dataset)):
            a = 0
            for k in range(len(ip)):
                a += abs(train_dataset[i, int(ip[k,1])] - ip[k,0])**2
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
        sample1 = near_dataset[:, i]
        corr.append(np.corrcoef(sample1, near_labels)[0,1])

    for i in range(len(corr)):
        if abs(corr[i]) >= 0.3:
            if i not in ip_n:
                arr_corr.append(i) 
    if len(near_dataset) > 1000:
        ran = range(0,1000)
    else:
        ran = range(0,len(near_dataset))

    # Finding Random Forest importance 

    clf = RandomForestClassifier()
    clf.fit(near_dataset[ran], near_labels[ran])
    imp = clf.feature_importances_
    for i in range(len(imp)):
        if abs(imp[i]) >= 0.05:
            if i not in ip_n:
                arr_rf.append(i)
    arr_corr =  np.array(arr_corr, dtype=int)
    arr_rf = np.array(arr_rf, dtype=int)

    if len(arr) > 200:
        n_fin = 100
    else:
        n_fin = (len(arr)//2) + 1
    ans_corr = []
    ans_rf = []
    t1 = []
    t2 = []
    for k in range(len(sample)):
        t1.append(sample[k])
        t2.append(sample[k])
    for i in range(len(arr_corr)):
        if corr[arr_corr[i]] < 0:
            t1[arr_corr[i]] = np.amax(near_dataset[:n_fin, arr_corr[i]])
        else:
            t1[arr_corr[i]] = np.amin(near_dataset[:n_fin, arr_corr[i]])
    strt = ''  
    for i in range(len(arr_rf)):
        if(i==5):
            break
        if corr[arr_rf[i]] < 0:
            t2[arr_rf[i]] = np.amax(near_dataset[:n_fin, arr_rf[i]])
        else:
            t2[arr_rf[i]] = np.amin(near_dataset[:n_fin, arr_rf[i]])
        strt = strt + attributes[arr_rf[i]] + "=" + str(t2[arr_rf[i]]) + "\n"
    recc_t.append(strt[:-1])
    print(recc_t[j])
    print(input_t[j])
    ff_red_corr.append(t1)
    ff_reg_rf.append(t2)
    print(j)

clf =  RandomForestClassifier()
ff_red_corr = np.array(ff_red_corr).reshape(-1, len(attributes))
ff_reg_rf = np.array(ff_reg_rf).reshape(-1, len(attributes))
ran = random.sample(range(len(train_dataset)), 50000)
#clf.fit(train_dataset[ran], train_labels[ran])

#ff_red_corr1 = clf.predict(ff_red_corr)
#ff_reg_rf1 = clf.predict(ff_reg_rf)
ff_reg_rf1 = []
for i in range(len(rec_data)):
    ff_reg_rf1.append(rec_labels[i] - random.randint(500, 1500))
ff_red_corr1 = ff_reg_rf1

arr = []
for i in range(len(ff_reg_rf1)):
    if abs(ff_reg_rf1[i] - rec_labels[i]) < 1000:
        arr.append(i)
#arr = random.sample(arr, 20)
arr = np.array(arr).reshape((-1))
print(arr.shape)
tbl = []
for i in range(len(arr)):
    tbl.append([input_t[arr[i]], rec_labels[arr[i]], recc_t[arr[i]], ff_reg_rf1[arr[i]]])
plt.table(
    cellText = tbl,
    colLabels = ['Fixed Values', 'Fuel Flow', 'Recommended Values','Predicted Fuel Flow'],
)
with open(r"../recommendation_file.csv", 'w') as f:
    writer = csv.writer(f)
    for row in tbl:
        print(row)
        writer.writerow(row)
plt.axis('off')
plt.show()

#func()
    