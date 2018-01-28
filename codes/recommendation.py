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

# Reading and Processing input
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
limt = m.sqrt(0.01*len(ip))
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

print(order)

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
    ran = random.sample(range(len(near_dataset)), 1000)
else:
    ran = range(0,len(near_dataset))

# Finding Random Forest importance 

clf = RandomForestClassifier()
clf.fit(near_dataset[ran], near_labels[ran])
imp = clf.feature_importances_
print("\nImportant Variables from random forest:")
for i in range(len(imp)):
    if abs(imp[i]) >= 0.048:
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

print("\nRecommendations based on correlation model :")
for i in range(len(arr_corr)):
    ans_corr.append(np.sum(near_dataset[:n_fin, arr_corr[i]])/n_fin)
    ans_corr[i] = std[arr_corr[i]]*ans_corr[i] + mean[arr_corr[i]]
    if corr[arr_corr[i]] < 0:
        print(attributes[arr_corr[i]], ">", ans_corr[i])
    else:
        print(attributes[arr_corr[i]], "<", ans_corr[i])
print("\nRecommendations based on Random Forest Model : ")
for i in range(len(arr_rf)):
    ans_rf.append(np.sum(near_dataset[:n_fin, arr_rf[i]])/n_fin)
    ans_rf[i] = std[arr_rf[i]]*ans_rf[i] + mean[arr_rf[i]]
    if corr[arr_rf[i]] < 0:
        print(attributes[arr_rf[i]], ">", ans_rf[i])
    else:
        print(attributes[arr_rf[i]], "<", ans_rf[i])


path = r"..\reco_plots\plot_"
for i in range(len(attributes)):
    plt.clf()
    points = near_dataset[:n_fin, i]
    plt.plot(points, near_labels[:n_fin], 'ro')
    plt.ylabel('Fuel Flow')
    plt.xlabel(str(i) + attributes[i])
    plt.savefig(path + str(i) + "_" + attributes[i] + "clean.png")






