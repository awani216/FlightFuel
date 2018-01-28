import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn import tree
import random
import math as m

# Opening data
with open(r"../data_reduction_2.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']
ran = np.random.permutation(len(train_labels))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]
ran = random.sample(range(len(train_dataset)),50000)
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]

clf = RandomForestRegressor()
clf.fit(train_dataset, train_labels)

imp = clf.feature_importances_
temp = []
for i in range(len(attributes)):
    temp.append([imp[i], attributes[i]])

temp.sort()
temp = np.array(temp)
print(temp)
imp.sort()
for i in range(len(attributes)):
    print(attributes[i],' ',imp[i])
plt.plot(imp,np.arange(1,len(attributes)+1),'black')
plt.yticks(np.arange(1,len(attributes)+1), temp[:,1])
plt.show()