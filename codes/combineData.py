# Created a pickle file for containing data for later use
# Removed 7 columns : X, ACID, Flight Instance, Year, Minute, Second 

import numpy as np
import csv
from six.moves import cPickle as pickle

path = r"..\Clean_Data\Train\takeoff"

train_dataset=[]
train_labels=[]
att_name = []

for i in range(5):
    with open(path+str(i+1)+".csv", 'r') as ip:
        fl = csv.reader(ip)
        for rows in fl :
            if rows[0] == '':
                continue
            train_dataset.append(rows[6:9] + rows[11:-1])
            train_labels.append(rows[-1])

test_dataset = []

path = r"..\Clean_Data\Test\takeoff.csv"

with open( path , 'r' ) as ip:
    fl = csv.reader(ip)
    for rows in fl :
        if rows[0] == '':
            att_name = (rows[5:8] + rows[10:-2])
            continue
        test_dataset.append(rows[5:8] + rows[10:-2])

picklefile = r"..\data_pickled.pickle"


train_dataset = np.array(train_dataset).astype(float)
train_labels = np.array(train_labels).astype(float)
test_dataset = np.array(test_dataset).astype(float)


try:
  f = open(picklefile, 'wb')
  data = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'test_dataset': test_dataset,
    'attributes': att_name
    }
  pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', picklefile, ':', e)
  raise

