import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np

with open(r"../data_reduction_1.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

path = r"..\final_plots\plot_"
#for i in range(len(attributes)):
plt.clf()
points = train_dataset[:,74]
plt.plot(points, train_labels,  'ko', markersize = 1)
plt.ylabel('Fuel Flow')
plt.xlabel(str(74) + attributes[74])
plt.savefig(path + str(74) + "_" + attributes[74] + "clean.png")
plt.show()
print("plotting graph " + str(74) + ": " + attributes[74])

