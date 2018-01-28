# 1. Removed columns with low variance
# 2. scaled the data to zero mean and unit variance


from six.moves import cPickle as pickle
import numpy as np
from sklearn import preprocessing

with open(r"../data_pickled.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes = data['attributes']
var = train_dataset.std(axis=0)
mean = train_dataset.mean(axis=0)


count = 0
for i in range(len(var)):
    if abs(mean[i]) > 0:
        if abs(var[i]/mean[i]) < 0.01:
            train_dataset = np.delete(train_dataset, i - count, 1)
            test_dataset = np.delete(test_dataset, i - count, 1)
            attributes = np.delete(attributes, i - count)
            count += 1

train_dataset = preprocessing.scale(train_dataset)
test_dataset = preprocessing.scale(test_dataset)

picklefile = r"..\data_scaled.pickle"
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