import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import random

with open(r"../data_reduction_3.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']
train_labels = np.array(train_labels, dtype=int)
clf = RandomForestClassifier()
#print(train_dataset.shape, train_labels.shape)
ran = random.sample(range(len(train_dataset)), 50000)

clf.fit(train_dataset[ran], train_labels[ran])
importance = clf.feature_importances_
res = []
for i in range(len(importance)):
    res.append([importance[i], i])

res.sort()
arr = []
imp = []

for i in range(len(res)):
    print("The importance of ", attributes[i], " is ", importance[i])
    if importance[i] <= 0.004:
        arr.append(i)
    else :
        imp.append(importance[i])

for i in range(3):
    arr.append(i)
    imp.pop()
arr.sort()
deleted = []
count = 0
for i in range(len(arr)):
    deleted.append(attributes[arr[i]])
    count+=1
for i in range(len(arr)):
    print(deleted[i], attributes[arr[i]-i])
    attributes.pop(arr[i] - i)
    train_dataset = np.delete(train_dataset, arr[i]-i, axis=1)
    test_dataset = np.delete(test_dataset, arr[i]-i, axis=1)   

print(len(train_dataset[0]))
print(attributes)

for i in range(len(attributes)):
    print(attributes[i],imp[i])

print(len(attributes))
print(len(imp))
 

print(count)

temp = []
for i in range(len(attributes)):
    temp.append([imp[i], attributes[i]])

temp.sort()
temp = np.array(temp)
print(temp)
imp.sort()
for i in range(len(attributes)):
    print(attributes[i],' ',imp[i])
plt.plot(np.arange(1,len(attributes)+1),imp,'black')
plt.xticks(np.arange(1,len(attributes)+1), temp[:,1],rotation = 'vertical')
plt.show()

picklefile = r"..\random_forest.pickle"
try:
    f = open(picklefile, 'wb')
    data = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'test_dataset': test_dataset,
        'attributes': attributes
        }
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', picklefile, ':', e)
    raise

path = "../reports/random_forest_reduction.txt"
with open(path, 'w') as f:
    f.write("Deleted columns in this step :" + str(len(deleted)) + "\n")
    for i in range(len(deleted)):
        f.write(deleted[i] + '\n')
    f.write("Remaininng columns : " + str(len(train_dataset[0])) + "\n")