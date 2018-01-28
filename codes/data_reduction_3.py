#Remove highly correlated data

import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np

with open(r"../data_reduction_2.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

corrmat = np.corrcoef(train_dataset, rowvar=False)
corrmat = abs(corrmat)
print(corrmat)
matyes = corrmat>0.85
matyes =np.array(matyes)
print(matyes.shape)

ind = []
count = 0
for i in range(len(matyes)):
    if i not in ind:
        count += 1
        for j in range(i+1, len(matyes)):
            if corrmat[i][j] > 0.8:
                if j not in ind:
                    ind.append(j)
ind.sort()
arr = ind
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

print(count)

picklefile = r"..\data_reduction_3.pickle"
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

path = "../reports/data_reduction_3.txt"
with open(path, 'w') as f:
    f.write("Deleted columns in this step :" + str(len(deleted)) + "\n")
    for i in range(len(deleted)):
        f.write(deleted[i] + '\n')
    f.write("Remaininng columns : " + str(len(train_dataset[0])) + "\n")