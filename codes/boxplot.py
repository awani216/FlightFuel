# boxplot
# whichever attribute you want to plot, change i to its corresponding data column number


import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np

with open(r"../data_pickled.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']


path = r"..\box_plot\plot_"
mean = np.mean(train_dataset,axis=0)
std = np.std(train_dataset, axis=0)
for i in range(len(train_dataset[0])):
    points = train_dataset[:,i]
    plt.clf()
    plt.figure()
    r = plt.boxplot(points)
    plt.savefig(path + attributes[i] + "_clean.png")
    if len(r["fliers"][0].get_data()[1]) <= 50:
        count = 0
        lim_dn = r["caps"][0].get_data()[1][0]
        lim_up = r["caps"][1].get_data()[1][0]
        for j in range(len(train_dataset)):
            if train_dataset[j,i] > lim_up:
                train_dataset[j,i] = mean[i]
                count += 1
            if train_dataset[j,i] < lim_dn:
                train_dataset[j,i] = mean[i]
                count += 1
        print("BoxPlot ", count)
        continue
    else:
        count = 0
        for j in range(len(train_dataset)):        
            if abs(train_dataset[j,i] - mean[i])/std[i] > 20:
                train_dataset[j,i] = mean[i]
                count += 1
        print("STD Dev ", count)

for i in range(len(train_dataset[0])):
    plt.clf()
    plt.figure()
    plt.boxplot(points)
    plt.savefig(path + str(i) + "_" + attributes[i] + "_clean.png")
    
    #print(r["caps"][0].get_data()[1][0],' ',r["caps"][1].get_data()[1][0],' ', len(r["fliers"][0].get_data()[1]))         
    #plt.show()
    #plt.savefig(path + str(i) + "_" + attributes[i] + ".png")
    #print("plotting graph " + str(i) + ": " + attributes[i])

points = train_labels
plt.clf()
plt.figure()
r = plt.boxplot(points)
mean = np.mean(train_labels)
std = np.std(train_labels)
if len(r["fliers"][0].get_data()[1]) <= 100:
    count = 0
    lim_dn = r["caps"][0].get_data()[1][0]
    lim_up = r["caps"][1].get_data()[1][0]
    for j in range(len(train_dataset)):
        if train_labels[j] > lim_up:
            train_labels[j] = mean
            count += 1
        if train_labels[j] < lim_dn:
            train_labels[j] = mean
            count += 1
    print("BoxPlot ", count)
else:
    count = 0
    
    for j in range(len(train_dataset)):        
        if abs(train_labels[j] - mean)/std > 20:
            train_labels[j] = mean
            count += 1
    print("STD Dev ", count)

picklefile = r"..\outliers_removed.pickle"
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
