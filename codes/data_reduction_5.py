import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np

with open(r"../data_reduction_4.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

arr = [0,1,11,21,22,23,24,26,4,42,5,8]
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

picklefile = r"..\data_reduction_5.pickle"
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